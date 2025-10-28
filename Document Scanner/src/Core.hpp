// docscanner_core.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

namespace DocScanner {

    static double euclidDist(const Point2f& a, const Point2f& b) {
        double dx = a.x - b.x, dy = a.y - b.y;
        return sqrt(dx * dx + dy * dy);
    }

    vector<Point2f> reorderPoints(const vector<Point2f>& pts) {
        if (pts.size() != 4) return pts;
        vector<Point2f> sorted = pts;
        sort(sorted.begin(), sorted.end(), [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
        vector<Point2f> left = { sorted[0], sorted[1] };
        vector<Point2f> right = { sorted[2], sorted[3] };
        sort(left.begin(), left.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });
        sort(right.begin(), right.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });
        return { left[0], right[0], right[1], left[1] };
    }

    Mat preProcessForContours(const Mat& img) {
        Mat gray, blurred, edges, closed;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        bilateralFilter(gray, blurred, 9, 75, 75);

        Mat flat = blurred.reshape(0, 1);
        flat.convertTo(flat, CV_8U);
        vector<uchar> vals(flat.datastart, flat.dataend);
        double med = 128;
        if (!vals.empty()) {
            nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
            med = vals[vals.size() / 2];
        }

        double lower = max(0.0, (1.0 - 0.33) * med);
        double upper = min(255.0, (1.0 + 0.33) * med);
        Canny(blurred, edges, lower, upper);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(edges, closed, MORPH_CLOSE, kernel);
        GaussianBlur(closed, closed, Size(3, 3), 0);
        return closed;
    }

    bool findDocumentContour(const Mat& pre, vector<Point2f>& outQuad) {
        vector<vector<Point>> contours;
        findContours(pre, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        if (contours.empty()) return false;

        sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a, false) > contourArea(b, false);
            });

        for (const auto& c : contours) {
            double area = contourArea(c);
            if (area < 1000) break;
            double peri = arcLength(c, true);
            vector<Point> approx;
            approxPolyDP(c, approx, 0.02 * peri, true);
            if (approx.size() == 4 && isContourConvex(approx)) {
                outQuad.clear();
                for (auto& p : approx) outQuad.push_back(Point2f(p));
                return true;
            }
        }

        // fallback: bounding box
        for (const auto& c : contours) {
            double area = contourArea(c);
            if (area < 1000) continue;
            RotatedRect r = minAreaRect(c);
            Point2f pts[4]; r.points(pts);
            outQuad.assign(pts, pts + 4);
            return true;
        }
        return false;
    }

    Mat getWarpedA4(const Mat& imgOrig, const vector<Point2f>& srcPts, int targetHeight = 842) {
        if (srcPts.size() != 4) return Mat();
        double aspect = 210.0 / 297.0;
        int h = targetHeight;
        int w = (int)round(h * aspect);

        vector<Point2f> dst = {
            Point2f(0, 0),
            Point2f((float)(w - 1), 0),
            Point2f((float)(w - 1), (float)(h - 1)),
            Point2f(0, (float)(h - 1))
        };

        Mat M = getPerspectiveTransform(srcPts, dst);
        Mat warped;
        warpPerspective(imgOrig, warped, M, Size(w, h), INTER_LINEAR, BORDER_CONSTANT);
        return warped;
    }

    Mat makeBWScanEffect(const Mat& warped) {
        Mat gray, clahe, bw;
        if (warped.channels() == 3)
            cvtColor(warped, gray, COLOR_BGR2GRAY);
        else
            gray = warped;

        Ptr<CLAHE> c = createCLAHE(2.0, Size(8, 8));
        c->apply(gray, clahe);
        adaptiveThreshold(clahe, bw, 255,
            ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 10);
        return bw;
    }

} // namespace DocScanner
