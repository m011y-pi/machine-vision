"""
Object Tracking with the Meanshift Algorithm -- with the math explained

Tracks a single object across video frames using meanshift. The object is
selected once at the start via a fixed bounding box (track_window), and its
hue histogram is used as a "fingerprint" to relocate it in each new frame
via back-projection.

Sample video: https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4

Controls while the preview window is focused:
    ESC -> quit
"""

import argparse
import numpy as np
import cv2 as cv


def track_with_meanshift(video_path: str) -> None:
    """Run meanshift object tracking on the given video file."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {video_path}")

    # Grab the first frame to initialize tracking
    ret, frame = cap.read()
    if not ret:
        raise IOError("Could not read the first frame from the video")

    # Initial location of the tracking window (hardcoded for this sample video)
    # (x, y) is the TOP-LEFT corner, w = width, h = height, in pixels.
    # The origin (0, 0) is the top-left of the image and y grows DOWNWARD.
    # So the window covers columns x .. x+w-1 and rows y .. y+h-1.
    x, y, w, h = 300, 200, 100, 50
    track_window = (x, y, w, h)

    # NumPy indexes images as [row, column] = [y, x], so the slice is
    # frame[y : y+h, x : x+w] -- rows first, then columns.
    roi = frame[y:y + h, x:x + w]

    # ------------------------------------------------------------------
    # WHY HSV INSTEAD OF BGR?
    # In BGR, a car in shadow and the same car in sunlight have totally
    # different (B, G, R) numbers. In HSV the COLOR TYPE is isolated in
    # one channel (Hue), separate from how vivid (Saturation) and how
    # bright (Value) it is. So hue alone is a lighting-robust fingerprint.
    #
    # OpenCV's 8-bit HSV ranges:
    #       H: 0..179   (real hue 0..360 degrees, halved to fit in a byte)
    #       S: 0..255
    #       V: 0..255
    # ------------------------------------------------------------------
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    # ------------------------------------------------------------------
    # MASK: only keep pixels whose hue is trustworthy.
    # cv.inRange(img, lower, upper) produces a binary mask:
    #       mask(p) = 255  if  lower <= pixel(p) <= upper  in ALL channels
    #       mask(p) = 0    otherwise
    # Here:  H in [0, 180],  S in [60, 255],  V in [32, 255].
    #
    # Why S >= 60 and V >= 32? Hue is mathematically UNDEFINED for gray
    # pixels (S = 0), and becomes pure noise when a pixel is very
    # desaturated (washed-out/white) or very dark (V near 0, where
    # camera noise dominates). Throwing those pixels out keeps garbage
    # out of the histogram.
    # ------------------------------------------------------------------
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    # ------------------------------------------------------------------
    # HUE HISTOGRAM of the object.
    # calcHist([img], [channel], mask, [bins], [range]):
    #       channel 0 = Hue,  180 bins,  range [0, 180)
    # => exactly one bin per integer hue value. The histogram is
    #       H[b] = number of (unmasked) ROI pixels whose hue == b
    # e.g. if the car is mostly red, H has big values near b = 0 and
    # b = 175..179 (red wraps around the hue circle) and ~0 elsewhere.
    # ------------------------------------------------------------------
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    # ------------------------------------------------------------------
    # Rescale the histogram to 0..255 (min-max normalization):
    #       H'[b] = 255 * (H[b] - H_min) / (H_max - H_min)
    # The most common hue becomes 255, the rarest becomes 0. Counts
    # depend on ROI size; this makes them a pure "how typical is this
    # hue for the object" score on the same scale as an 8-bit image,
    # which is what we need for the next step.
    # ------------------------------------------------------------------
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    # ------------------------------------------------------------------
    # TERMINATION CRITERIA for the meanshift iterations: stop when
    # EITHER of these happens first:
    #       - 10 iterations have run           (the COUNT condition)
    #       - the window center moves by less than 1 pixel (the EPS
    #         condition, i.e. it has converged)
    # The tuple is (type flags, max_iterations, epsilon).
    # ------------------------------------------------------------------
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frames grabbed - end of video or read error.")
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # --------------------------------------------------------------
        # HISTOGRAM BACK-PROJECTION
        # For EVERY pixel p in the new frame, look up its hue in the
        # object's normalized histogram:
        #       dst(p) = H'[ hue(p) ]
        # The output `dst` is a grayscale "probability image":
        #       bright pixel = its hue is common in the object
        #       dark pixel   = its hue never appeared in the object
        # So `dst` is a heat map of "how object-like is this pixel?".
        # (The final argument, scale = 1, multiplies the lookup value;
        # 1 leaves it unchanged.)
        #
        # Caveat: unlike the ROI step, no saturation/value mask is applied
        # to the new frame, so gray/dark pixels still get *some* score
        # based on their (meaningless) hue.
        # --------------------------------------------------------------
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # --------------------------------------------------------------
        # MEANSHIFT: climb the heat map to the nearest peak.
        #
        # Treat dst(x, y) as a "mass" at each pixel. Using only the
        # pixels currently inside the window, compute the image moments:
        #       M00 = sum  dst(x, y)              (total mass)
        #       M10 = sum  x * dst(x, y)          (x-weighted mass)
        #       M01 = sum  y * dst(x, y)          (y-weighted mass)
        # The CENTER OF MASS (centroid) of the window contents is:
        #       x_c = M10 / M00        y_c = M01 / M00
        # i.e. a weighted average of pixel coordinates, where brighter
        # (more object-like) pixels count more.
        #
        # Then: move the window so its center sits on (x_c, y_c), and
        # REPEAT. Each step shifts the window toward the denser region
        # ("the mean shifts"). It stops when the shift is < 1 px or 10
        # iterations pass (term_crit above). Because the window is
        # always pulled uphill, it follows the object as it moves a
        # short distance between frames.
        #
        # Notes:
        #   * The window SIZE never changes (CamShift is the variant that
        #     also adapts size and rotation).
        #   * Window center = (x + w/2, y + h/2).
        #   * The first return value (`ret`, number of iterations used)
        #     is unused here.
        #   * If the object jumps farther than the window can reach in
        #     one frame, there is no mass inside the window to climb
        #     and tracking is lost.
        # --------------------------------------------------------------
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # Draw the updated tracking window on the frame.
        # Rectangle corners: top-left (x, y), bottom-right (x+w, y+h).
        # The color `255` is a scalar, so on a 3-channel BGR image it
        # means (B, G, R) = (255, 0, 0) -> BLUE, not white. Thickness 2 px.
        x, y, w, h = track_window
        tracked_frame = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv.imshow("Meanshift Tracking", tracked_frame)

        # waitKey(30): wait up to 30 ms (~33 fps cap); `& 0xff` keeps the
        # low 8 bits (plain key code). ESC = 27.
        key = cv.waitKey(30) & 0xff
        if key == 27:  # ESC key - quit
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample demonstrates the meanshift algorithm. "
                     "The example file can be downloaded from: "
                     "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"
    )
    parser.add_argument("image", type=str, help="path to video file")
    args = parser.parse_args()

    track_with_meanshift(args.image)
