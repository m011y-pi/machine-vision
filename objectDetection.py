"""
Background Subtraction Demo (MOG2 / KNN) -- with the math explained

Applies OpenCV's background subtraction algorithms to a video to separate
moving foreground objects from a static background. Every frame is annotated
with its frame number, and the foreground mask is optionally saved to an
output video (inverted, so foreground objects appear dark on a light
background).

Controls while the preview window is focused:
    q or ESC -> quit
"""

from __future__ import print_function
import argparse
import cv2 as cv


def run_background_subtraction(input_path: str, algo: str, output_path: str | None) -> None:
    """Run background subtraction on `input_path` using the given algorithm."""

    # ==========================================================================
    # CHOOSING THE BACKGROUND MODEL
    # ==========================================================================
    #
    # Both algorithms keep a *per-pixel* statistical model of "what this pixel
    # usually looks like". A new pixel value that doesn't fit the model is
    # called foreground. The difference is HOW each one models a pixel.
    #
    # --------------------------------------------------------------------------
    # MOG2 = "Mixture of Gaussians, version 2" (Zivkovic 2004)
    # --------------------------------------------------------------------------
    # Every pixel is described by a mixture of K Gaussian bell curves (one
    # color channel triple x = (B, G, R)):
    #
    #       p(x) = sum_{k=1..K}  w_k * N(x ; mu_k, sigma_k^2 * I)
    #
    #   w_k        weight of component k  (how often this "look" occurs;
    #              all weights are >= 0 and sum to 1)
    #   mu_k       mean color of component k
    #   sigma_k^2  variance of component k (one shared variance for all 3
    #              channels, hence "* I", the identity matrix)
    #   N(...)     the Gaussian density:
    #                  N(x; mu, s^2 I) = (2*pi*s^2)^(-3/2)
    #                                    * exp( -||x - mu||^2 / (2*s^2) )
    #
    # Why a MIXTURE? Because one pixel can legitimately have several
    # background appearances: e.g. a leaf swaying in front of the sky gives
    # a "green" component and a "blue" component.
    #
    # MATCHING a new pixel x to a component (the "is it close enough?" test):
    #
    #       ||x - mu_k||^2 / sigma_k^2  <  T          (T = varThreshold)
    #
    #   The left side is the squared Mahalanobis distance: "how many standard
    #   deviations away is x, squared?". OpenCV's default T = 16, which means
    #   sqrt(16) = 4 standard deviations. In a Gaussian, ~99.99% of samples
    #   fall within 4 sigma, so anything beyond is very unlikely to be
    #   "normal background".
    #
    # UPDATING the model online (runs every frame, so the model adapts):
    #
    #   alpha = 1 / history      -> learning rate. Default history = 500,
    #                               so alpha = 0.002: each new frame
    #                               contributes 0.2% to the model. This is an
    #                               exponential moving average (EMA) with an
    #                               effective memory of about `history` frames.
    #
    #   Weights:   w_k <- (1 - alpha) * w_k + alpha * M_k
    #       M_k = 1 for the component that matched, 0 for the others.
    #       Matched components grow, unmatched ones decay; weights are then
    #       renormalized so they sum to 1 again.
    #
    #   Mean:      mu_k <- (1 - rho) * mu_k + rho * x
    #   Variance:  sigma_k^2 <- (1 - rho) * sigma_k^2 + rho * ||x - mu_k||^2
    #       where rho = alpha / w_k  (Zivkovic's version; the older MOG
    #       used rho = alpha * N(x | mu_k, sigma_k)).
    #       Both are again EMAs, just nudging the Gaussian toward the sample.
    #
    #   If NO component matches, the least-important component is thrown
    #   away and replaced by a new Gaussian centered on x with a large
    #   initial variance and a small initial weight (a "new appearance").
    #
    # DECIDING which components are BACKGROUND:
    #   Sort the components by w_k / sigma_k (heavy and tight = reliable),
    #   then take the first B components whose cumulative weight exceeds
    #   the background ratio (default 0.9):
    #
    #       B = argmin_b  ( sum_{k=1..b} w_k  >  0.9 )
    #
    #   A pixel that matches one of those first B components -> background.
    #   Anything else -> foreground.
    #
    # SHADOWS (detectShadows=True by default):
    #   A pixel is flagged as a shadow if it is the same *color direction*
    #   as the background but darker, i.e. x ~= c * mu with 0.5 < c < 1
    #   (threshold 0.5 by default). Shadows get the gray mask value 127,
    #   which is why the mask has THREE values: 0, 127, 255.
    #
    # --------------------------------------------------------------------------
    # KNN = "K-Nearest Neighbors" background subtractor (Zivkovic 2006)
    # --------------------------------------------------------------------------
    # No Gaussians at all. Instead each pixel stores a small bag of recent
    # raw color samples {s_1, s_2, ..., s_N}, and a new pixel x is judged
    # with a non-parametric (kernel-free) density estimate:
    #
    #       count = #{ i : ||x - s_i||^2 < dist2Threshold }
    #
    #   ||x - s_i||^2 is the squared Euclidean distance in color space:
    #       (B - B_i)^2 + (G - G_i)^2 + (R - R_i)^2
    #   dist2Threshold defaults to 400, i.e. a color distance of
    #   sqrt(400) = 20 intensity levels.
    #
    #   If `count` is at least k (the "kNN samples" setting), enough stored
    #   samples agree with x -> background. Otherwise -> foreground.
    #
    #   The argument below (100) is `history`: how many frames of memory.
    #   The sample bag is refreshed at a rate of roughly 1/history per
    #   frame, so a smaller history (100 vs the default 500) makes the
    #   model forget faster and adapt more quickly to scene changes.
    #
    # TL;DR: MOG2 summarizes a pixel's past with a few bell curves (cheap,
    # smooth, assumes roughly Gaussian noise); KNN keeps raw samples and
    # asks "have I seen something like this before?" (more flexible for
    # odd, multi-modal backgrounds, a bit slower).
    # ==========================================================================
    if algo == "MOG2":
        back_sub = cv.createBackgroundSubtractorMOG2()
    else:
        back_sub = cv.createBackgroundSubtractorKNN(100)

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(input_path))
    if not capture.isOpened():
        raise IOError(f"Unable to open: {input_path}")

    # Set up the output video writer only if an output path was supplied
    out_writer = None
    if output_path:
        # Frame size in pixels: (width, height). Note OpenCV's VideoWriter
        # wants (width, height) but NumPy image arrays are (rows, cols) =
        # (height, width) -- an easy mix-up to make.
        size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv.VideoWriter_fourcc(*"mp4v")

        # fps = 25 means each frame lasts 1/25 s = 40 ms in the saved file.
        out_writer = cv.VideoWriter(output_path, fourcc, 25.0, size)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        # ----------------------------------------------------------------------
        # Run the model on this frame. Two things happen in one call:
        #   1. CLASSIFY every pixel using the rules described above
        #      (match test for MOG2, neighbor count for KNN).
        #   2. UPDATE the per-pixel model with this frame (the EMA updates
        #      with learning rate alpha ~ 1/history).
        #
        # Output: a single-channel uint8 image, same height/width as `frame`:
        #       0   = background
        #       127 = shadow (only if shadow detection is on)
        #       255 = foreground
        # ----------------------------------------------------------------------
        fg_mask = back_sub.apply(frame)

        # ----------------------------------------------------------------------
        # Frame counter overlay.
        #
        # cv.rectangle(img, top_left, bottom_right, color, thickness):
        #   Corners are (x, y) with the origin at the TOP-LEFT of the image
        #   and y growing DOWNWARD. (10, 2) -> (100, 20) is a box 90 px wide
        #   and 18 px tall. Colors are (B, G, R), so (255, 255, 255) is
        #   white. thickness = -1 means "fill the shape".
        #
        # CAP_PROP_POS_FRAMES is the 0-based index of the NEXT frame to be
        # read. Because we've just read a frame, this value already equals
        # (index of the current frame) + 1, so the label shows the familiar
        # 1-based frame number: 1, 2, 3, ...
        # ----------------------------------------------------------------------
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fg_mask)

        if out_writer is not None:
            # ------------------------------------------------------------------
            # fg_mask is 1 channel (H x W) but the video file expects 3
            # channels (H x W x 3). GRAY2RGB just copies the gray value into
            # all three channels:
            #       (g)  ->  (g, g, g)
            # so a mask value of 255 becomes (255, 255, 255), pure white.
            #
            # bitwise_not flips every bit of every 8-bit value, which for
            # uint8 is exactly:
            #       v' = 255 - v
            #   e.g. 255 (foreground) -> 0   (black)
            #        127 (shadow)     -> 128 (mid gray)
            #          0 (background) -> 255 (white)
            # So the saved video shows dark objects on a white background,
            # which is nicer to read on paper or in slides.
            # ------------------------------------------------------------------
            mask_bgr = cv.cvtColor(fg_mask, cv.COLOR_GRAY2RGB)
            mask_bgr = cv.bitwise_not(mask_bgr)
            out_writer.write(mask_bgr)

        # ----------------------------------------------------------------------
        # waitKey(30): wait up to 30 ms for a keypress. That caps playback
        # at about 1000 / 30 ~= 33 frames per second (less if processing is
        # slow). Returns -1 if no key was pressed.
        #
        # `& 0xff` keeps only the lowest 8 bits. On some platforms waitKey
        # returns extra high bits (modifier/flag bits), so masking leaves
        # just the plain ASCII key code:
        #       'q' = 113, ESC = 27
        # ord("q") converts the character to that integer for comparison.
        # (-1 & 0xff = 255, which matches neither, so "no key" is ignored.)
        # ----------------------------------------------------------------------
        keyboard = cv.waitKey(30) & 0xff
        if keyboard == ord("q") or keyboard == 27:  # 'q' or ESC
            break

    capture.release()
    if out_writer is not None:
        out_writer.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program shows how to use background subtraction methods "
                     "provided by OpenCV. You can process both videos and images."
    )
    parser.add_argument("--input", type=str, default="vtest.avi",
                         help="Path to a video or a sequence of images.")
    parser.add_argument("--algo", type=str, default="MOG2", choices=["MOG2", "KNN"],
                         help="Background subtraction method (KNN, MOG2).")
    parser.add_argument("--output", type=str, default=None,
                         help="If supplied, path to save the foreground mask video.")
    args = parser.parse_args()

    run_background_subtraction(args.input, args.algo, args.output)
