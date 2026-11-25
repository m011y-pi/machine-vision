"""
Dense Optical Flow Visualization (Farneback method) -- with the math explained

Reads a video and computes the dense optical flow between consecutive frames
using Gunnar Farneback's algorithm. The flow field is visualized as an HSV
image where:
    - Hue        -> direction of motion (angle of the flow vector)
    - Saturation -> fixed at max (255)
    - Value      -> magnitude of motion (normalized brightness)

Controls while the preview window is focused:
    ESC -> quit
    s   -> save the current frame and its flow visualization to disk
"""

import numpy as np
import cv2 as cv


def compute_optical_flow(video_path: str) -> None:
    """Run dense optical flow visualization on the given video file."""
    cap = cv.VideoCapture(cv.samples.findFile(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {video_path}")

    # Read the first frame to use as our initial "previous" frame
    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("Could not read the first frame from the video")

    # ------------------------------------------------------------------
    # Optical flow works on BRIGHTNESS only, so we throw away color.
    # OpenCV's BGR -> gray conversion uses the standard luma weights:
    #       Y = 0.299 * R + 0.587 * G + 0.114 * B
    # (green gets the biggest weight because our eyes are most sensitive
    # to it). Result: one uint8 value (0..255) per pixel.
    # ------------------------------------------------------------------
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    # HSV image used for visualizing flow; same size as the input frames.
    # np.zeros_like gives an H x W x 3 uint8 array of zeros.
    hsv = np.zeros_like(prev_frame)
    # Channel 1 = Saturation. Fixing it at 255 means every color is fully
    # vivid, so only Hue (direction) and Value (speed) carry information.
    hsv[..., 1] = 255

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frames grabbed - end of video or read error.")
            break

        curr_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # ==================================================================
        # THE MATH OF FARNEBACK OPTICAL FLOW
        # ==================================================================
        #
        # GOAL: for every pixel (x, y), find a motion vector (u, v) = how
        # many pixels it moved right (u) and down (v) between the two
        # frames. "Dense" means we get a vector for EVERY pixel, not just
        # a few corner points.
        #
        # --- Step 0: the brightness-constancy assumption ----------------
        # Assume a moving point keeps the same brightness:
        #       I(x + u, y + v, t + 1) = I(x, y, t)
        # A first-order Taylor expansion turns this into the classic
        # "optical flow constraint equation":
        #       Ix * u + Iy * v + It = 0
        # where Ix, Iy are the spatial brightness gradients and It is the
        # change in brightness over time. That is ONE equation with TWO
        # unknowns (u, v) per pixel -- the "aperture problem": looking
        # through a small window at a straight edge, you can only tell
        # how far it moved perpendicular to the edge. Every optical-flow
        # method has to add extra assumptions to solve this. Farneback's
        # trick is below.
        #
        # --- Step 1: polynomial expansion -------------------------------
        # Around every pixel, fit a quadratic surface to the brightness of
        # its neighborhood (weighted by a Gaussian):
        #       f(x) ~= x^T A x + b^T x + c
        #   x = (x, y)^T is the position relative to the center pixel,
        #   A is a 2x2 symmetric matrix (curvature of the brightness),
        #   b is a 2-vector (slope of the brightness),
        #   c is a scalar (brightness offset).
        # This is done for BOTH frames, giving (A1, b1, c1) and
        # (A2, b2, c2) at every pixel.
        #
        # --- Step 2: what a shift does to the polynomial ----------------
        # If frame 1's neighborhood is f1(x) = x^T A1 x + b1^T x + c1 and
        # the signal simply moves by a displacement d, then
        #       f2(x) = f1(x - d)
        #             = x^T A1 x + (b1 - 2 * A1 * d)^T x + (...constant)
        # Matching coefficients with f2(x) = x^T A2 x + b2^T x + c2 gives
        #       A2 = A1                       (curvature doesn't change)
        #       b2 = b1 - 2 * A1 * d
        # Solve for d (a closed-form, no iteration needed per pixel!):
        #       d = -(1/2) * A1^(-1) * (b2 - b1)
        # In practice A1 and A2 differ slightly because of noise, so
        # OpenCV uses their average: A = (A1 + A2) / 2.
        #
        # --- Step 3: smoothing over a window ----------------------------
        # A single pixel's estimate of d is noisy. So instead of solving
        # per pixel, we solve in a least-squares sense over a whole window
        # (size = `winsize`, Gaussian-weighted):
        #       minimize  sum_over_window  w(x) * || A(x) d - dB(x) ||^2
        #   where  dB(x) = -(1/2) * (b2(x) - b1(x)).
        # That's an ordinary 2x2 linear system per pixel, solved directly.
        # Averaging over the window is what resolves the aperture problem:
        # edges in different directions inside the window pin down both
        # components of d.
        #
        # --- Step 4: iterate + pyramid (handles BIG motions) ------------
        # The polynomial model only holds for SMALL shifts (about a pixel
        # or two). To handle bigger motion:
        #   * Pyramid: build a stack of images, each `pyr_scale` times the
        #     size of the one below (0.5 => each level is half the width
        #     and height). Motion that is 20 px at full size is only 10 px
        #     at the next level, 5 px at the next, etc. We estimate flow at
        #     the coarsest (smallest) level first, where the shift is tiny.
        #   * Warp & refine: upscale that flow (multiply vectors by
        #     1 / pyr_scale = 2), warp frame 2 back by it so the remaining
        #     motion is small again, and re-solve for the leftover. This
        #     refinement is repeated `iterations` times per level.
        #   With pyr_scale = 0.5 and levels = 3, the maximum motion that
        #   can be tracked is roughly (winsize-ish) * 2^(levels - 1).
        # ==================================================================
        flow = cv.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,   # `None` = allocate the output for us
            0.5,  # pyr_scale: each pyramid level is 0.5x the previous size
            3,    # levels: 3 levels total (original + 2 downscaled copies)
            15,   # winsize: 15x15 averaging window (Step 3). Bigger window
                  #          = smoother, more noise-robust, blurrier at edges
            3,    # iterations: warp-and-refine passes per pyramid level
            5,    # poly_n: polynomial expansion uses a 5x5 neighborhood
                  #          (Step 1). Typically 5 or 7.
            1.2,  # poly_sigma: std dev (pixels) of the Gaussian used to
                  #          weight/smooth that neighborhood. Rule of thumb:
                  #          1.1 for poly_n=5, 1.5 for poly_n=7.
            0     # flags: 0 = defaults
        )
        # `flow` has shape (H, W, 2) with float32 values:
        #       flow[..., 0] = u (horizontal motion, +ve = rightward)
        #       flow[..., 1] = v (vertical motion,   +ve = downward)
        # because image y-coordinates grow downward.

        # ------------------------------------------------------------------
        # CARTESIAN -> POLAR conversion
        # Each pixel has a vector (u, v). Convert to length and direction:
        #       magnitude = sqrt(u^2 + v^2)            (pixels per frame)
        #       angle     = atan2(v, u)                (radians)
        # cartToPolar returns the angle in [0, 2*pi): 0 = pointing right,
        # pi/2 = pointing down (remember y is down!), pi = left,
        # 3*pi/2 = up.
        # ------------------------------------------------------------------
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # ------------------------------------------------------------------
        # HUE: angle -> OpenCV's 8-bit hue scale
        # Normally hue is 0..360 degrees. But a uint8 only holds 0..255,
        # so OpenCV halves it: hue range is 0..179 (each unit = 2 degrees).
        # Conversion chain:
        #       radians -> degrees:   angle * 180 / pi        (0..360)
        #       degrees -> OpenCV:    ... / 2                 (0..180)
        # which is exactly  angle * 180 / pi / 2  =  angle * 90 / pi.
        # Quick check: angle = pi (pointing left)
        #       -> pi * 90 / pi = 90  (cyan-ish in OpenCV hue).
        # The result is a float array; storing into the uint8 hsv array
        # truncates it to an integer (always < 180, so no overflow).
        # ------------------------------------------------------------------
        hsv[..., 0] = angle * 180 / np.pi / 2

        # ------------------------------------------------------------------
        # VALUE (brightness): min-max normalize the magnitudes to 0..255.
        #       v' = 255 * (m - m_min) / (m_max - m_min)
        # The smallest motion in THIS frame maps to 0 (black), the largest
        # to 255 (full brightness). IMPORTANT: this is recomputed per frame,
        # so brightness is RELATIVE, not absolute -- a nearly-still frame
        # will still show its tiny largest motion as bright white. To
        # compare speeds across frames, use a fixed scale instead.
        # ------------------------------------------------------------------
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # HSV -> BGR so the screen can display it. Result: black = no
        # motion, bright colors = motion, with the color encoding direction.
        flow_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow("Optical Flow", flow_bgr)

        # waitKey(30): wait up to 30 ms (~33 fps cap). `& 0xff` keeps only
        # the low 8 bits (the plain ASCII key code), discarding any
        # platform-specific flag bits. ESC = 27.
        key = cv.waitKey(30) & 0xff
        if key == 27:  # ESC key - quit
            break
        elif key == ord("s"):  # 's' key - save current frame + flow image
            cv.imwrite("opticalfb.png", frame)
            cv.imwrite("opticalhsv.png", flow_bgr)
            print("Saved opticalfb.png and opticalhsv.png")

        # Slide the window forward: this frame becomes the "previous" one.
        prev_gray = curr_gray

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = input("Enter path to video file: ").strip()
    compute_optical_flow(video_path)
