"""
Foreground Extraction via Otsu Thresholding + Bitwise Composite
-- with the math explained

For each frame of a "foreground" video, this script:
    1. Converts the frame to grayscale.
    2. Uses Otsu's method to automatically threshold it into a mask
       (separating bright/dark regions - typically foreground vs. background).
    3. Uses that mask to invert (bitwise NOT) the masked pixels of the
       foreground frame directly onto a frame from a second "background"
       video, producing a composited output frame.

Controls while the preview window is focused:
    q -> quit
"""

import argparse
import cv2
import numpy as np


def process_videos(input_path: str, top_path: str, output_path: str | None) -> None:
    """Composite frames from `input_path` and `top_path` using an Otsu mask."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")

    cap_top = cv2.VideoCapture(top_path)
    if not cap_top.isOpened():
        raise IOError(f"Could not open top/background video: {top_path}")

    # Set up the output video writer, matching the input video's resolution
    out_writer = None
    if output_path:
        # VideoWriter wants the size as (width, height) in pixels -- the
        # reverse of NumPy's (rows, cols) = (height, width) array shape.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 25 fps => each frame lasts 1/25 s = 40 ms in the saved file.
        out_writer = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        ret_top, frame_top = cap_top.read()

        # Stop once either video runs out of frames
        if not ret or not ret_top:
            break

        cv2.imshow("Frame", frame)

        # ------------------------------------------------------------------
        # GRAYSCALE: collapse (B, G, R) to one brightness number per pixel
        # using the standard luma weights:
        #       Y = 0.299 * R + 0.587 * G + 0.114 * B
        # Green is weighted most because human vision is most sensitive to
        # it; blue least. Output is a uint8 image with values 0..255.
        # ------------------------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ==================================================================
        # OTSU'S METHOD -- automatically choosing the threshold
        # ==================================================================
        # Imagine the grayscale histogram: p(i) = fraction of pixels with
        # brightness i, for i = 0..255. Pick a threshold t and split the
        # pixels into two classes:
        #       class 0 ("dark"):    brightness <= t
        #       class 1 ("bright"):  brightness >  t
        #
        # Class probabilities (what fraction of pixels fall in each):
        #       w0(t) = sum_{i=0..t}   p(i)
        #       w1(t) = sum_{i=t+1..255} p(i) = 1 - w0(t)
        #
        # Class mean brightnesses:
        #       mu0(t) = ( sum_{i=0..t}     i * p(i) ) / w0(t)
        #       mu1(t) = ( sum_{i=t+1..255} i * p(i) ) / w1(t)
        #
        # BETWEEN-CLASS VARIANCE -- how far apart the two groups are,
        # weighted by how big each group is:
        #       sigma_b^2(t) = w0(t) * w1(t) * ( mu0(t) - mu1(t) )^2
        #
        # Otsu tries EVERY t from 0 to 255 and picks
        #       t* = argmax_t  sigma_b^2(t)
        # Intuition: the best threshold is the one that makes the two
        # groups as different from each other as possible. (It is
        # mathematically equivalent to minimizing the variance WITHIN each
        # group, since total variance = within + between is a constant.)
        # It works best when the histogram is bimodal (two humps), e.g.
        # a dark scene with a bright subject.
        #
        # The threshold value passed below (0) is IGNORED when the
        # THRESH_OTSU flag is set; Otsu computes t* itself per frame.
        # (Because t* is recomputed for every frame, the cut-off can
        # flicker if lighting or content changes.)
        #
        # THRESH_TOZERO rule (note: NOT a binary threshold!):
        #       mask(p) = gray(p)   if gray(p) > t*
        #       mask(p) = 0         otherwise
        # So bright pixels KEEP their gray value and dark pixels become 0.
        # The `+` combines the two flag bits (they are separate bit
        # flags, so adding is the same as OR-ing them).
        #
        # For the next step only "zero vs nonzero" matters, so this still
        # works as a mask: nonzero = "brighter than t*".
        # ==================================================================
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

        # ==================================================================
        # MASKED BITWISE NOT -- the actual compositing step
        # ==================================================================
        # For 8-bit pixels, bitwise NOT flips each of the 8 bits:
        #       ~v = 255 - v
        # Examples:  0b00000000 (0)   -> 0b11111111 (255)
        #            0b11110000 (240) -> 0b00001111 (15)
        # which turns every color channel into its "photographic negative".
        #
        # With a mask and an explicit destination, the rule is, per pixel p
        # (per channel):
        #       result(p) = 255 - frame(p)     if mask(p) != 0
        #       result(p) = frame_top(p)       if mask(p) == 0
        # (unmasked pixels are simply left as they already were in the
        # destination array, `frame_top`).
        #
        # So the output is: the background video, with the NEGATIVE of the
        # foreground video "punched through" wherever the foreground is
        # brighter than the Otsu threshold.
        #
        # Requirements: `frame`, `frame_top` and `mask` must all have the
        # same height and width, and `frame_top` the same type/channels as
        # `frame`. Also note the destination is modified IN PLACE: after
        # this call, `frame_top` and `composite` are the same array.
        # ==================================================================
        composite = cv2.bitwise_not(frame, frame_top, mask=mask)

        cv2.imshow("Composite", composite)
        if out_writer is not None:
            out_writer.write(composite)

        # waitKey(25): wait up to 25 ms for a key (~40 fps cap). `& 0xFF`
        # keeps only the low 8 bits (plain ASCII code); 'q' = 113.
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cap_top.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Composite two videos using an Otsu-thresholded bitwise mask."
    )
    parser.add_argument("--input", type=str, default="vtest.avi",
                         help="Path to the foreground video or image sequence.")
    parser.add_argument("--top", type=str, required=True,
                         help="Path to the background video to composite onto.")
    parser.add_argument("--output", type=str, default=None,
                         help="If supplied, path to save the composited output video.")
    args = parser.parse_args()

    process_videos(args.input, args.top, args.output)
