#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

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
    vector<Point2f> ordered(4);
    ordered[0] = left[0];   // tl
    ordered[1] = right[0];  // tr
    ordered[2] = right[1];  // br
    ordered[3] = left[1];   // bl
    return ordered;
}

// basic Canny preproc for auto detection
Mat preProcessForContours(const Mat& img) {
    Mat gray, blurred, edges, kernel, closed;
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
    kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
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

Mat getWarped(const Mat& imgOrig, const vector<Point2f>& srcPts) {
    if (srcPts.size() != 4) return Mat();
    auto widthA = euclidDist(srcPts[2], srcPts[3]);
    auto widthB = euclidDist(srcPts[1], srcPts[0]);
    auto maxWidth = max(widthA, widthB);
    auto heightA = euclidDist(srcPts[1], srcPts[2]);
    auto heightB = euclidDist(srcPts[0], srcPts[3]);
    auto maxHeight = max(heightA, heightB);
    int w = max(1, (int)round(maxWidth));
    int h = max(1, (int)round(maxHeight));
    vector<Point2f> dst = { Point2f(0,0), Point2f((float)(w - 1),0), Point2f((float)(w - 1),(float)(h - 1)), Point2f(0,(float)(h - 1)) };
    Mat M = getPerspectiveTransform(srcPts, dst);
    Mat warped;
    warpPerspective(imgOrig, warped, M, Size(w, h), INTER_LINEAR, BORDER_CONSTANT);
    return warped;
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

// ---------- Globals for interactive UI ----------
Mat imgOrig;            // original full-resolution image (coords stored in this space)
Mat imgDisplayScaled;   // what we show on screen (scaled)
Mat lastWarpColor;
double scaleFactor = 1.0; // imgDisplayScaled = resize(imgOrig, scaleFactor)
int MAX_WIN_W = 1200;   // default fit size - change to monitor resolution if you know it
int MAX_WIN_H = 800;

vector<Point2f> manualPts; // stored in original-image coordinates
vector<Point2f> autoPts;   // original-image coords (auto-detected)
bool manualMode = false;
bool dragging = false;
int dragIdx = -1;

// convert display (window) coords to original-image coords
Point2f dispToOrig(int x, int y) {
    // screen shows imgDisplayScaled with size (imgDisplayScaled.cols, imgDisplayScaled.rows)
    // mapping: orig_x = x / scaleFactor, orig_y = y / scaleFactor
    return Point2f((float)(x / scaleFactor), (float)(y / scaleFactor));
}

// convert original-image coords to display coords for drawing
Point dispFromOrig(const Point2f& p) {
    return Point((int)round(p.x * scaleFactor), (int)round(p.y * scaleFactor));
}

void drawOverlay(Mat& dst) {
    // draw auto-detected pts (blue)
    if (autoPts.size() == 4) {
        vector<Point> ap;
        for (auto& p : autoPts) ap.push_back(dispFromOrig(p));
        polylines(dst, ap, true, Scalar(255, 150, 0), 2);
        for (int i = 0; i < 4; i++) {
            circle(dst, dispFromOrig(autoPts[i]), 6, Scalar(255, 150, 0), FILLED);
            putText(dst, "A" + to_string(i + 1), dispFromOrig(autoPts[i]) + Point(5, -5), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 150, 0), 2);
        }
    }
    // draw manual pts (green)
    if (!manualPts.empty()) {
        if (manualPts.size() >= 2) {
            vector<Point> mp;
            for (auto& p : manualPts) mp.push_back(dispFromOrig(p));
            if (manualPts.size() == 4) polylines(dst, mp, true, Scalar(0, 200, 50), 2);
        }
        for (int i = 0; i < manualPts.size(); i++) {
            circle(dst, dispFromOrig(manualPts[i]), 8, Scalar(0, 200, 50), FILLED);
            putText(dst, to_string(i + 1), dispFromOrig(manualPts[i]) + Point(6, -6), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 1);
        }
    }
    string mode = manualMode ? "MANUAL MODE (m toggle)" : "AUTO MODE (m toggle)";
    putText(dst, mode, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 4);
    putText(dst, mode, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);
    putText(dst, "Left-click: add/select/drag | Right-click: remove nearest | a: copy auto->manual", Point(10, 50),
        FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 255), 1);
    putText(dst, "w: warp | s: save | r: reset manual | q: quit", Point(10, 70),
        FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 255), 1);
}

// mouse callback works in display coordinates (scaled)
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (!manualMode) return; // only editing in manual mode

    Point2f origPt = dispToOrig(x, y);

    if (event == EVENT_LBUTTONDOWN) {
        // find nearest manual point in orig-space
        int nearest = -1;
        double minD = 1e9;
        for (int i = 0; i < manualPts.size(); i++) {
            double d = euclidDist(origPt, manualPts[i]);
            if (d < minD) { minD = d; nearest = i; }
        }
        if (minD < 20.0 / scaleFactor && nearest >= 0) {
            // start dragging (threshold in orig-space adjusted by scale)
            dragging = true; dragIdx = nearest;
        }
        else {
            // add a point (in original coords) up to 4
            if (manualPts.size() < 4) {
                manualPts.push_back(origPt);
            }
            else {
                // if already 4, clicking near one will replace it (use a larger tolerance)
                if (nearest >= 0 && minD < 40.0 / scaleFactor) {
                    manualPts[nearest] = origPt;
                }
            }
        }
    }
    else if (event == EVENT_MOUSEMOVE) {
        if (dragging && dragIdx >= 0 && dragIdx < manualPts.size()) {
            manualPts[dragIdx] = origPt;
        }
    }
    else if (event == EVENT_LBUTTONUP) {
        dragging = false; dragIdx = -1;
    }
    else if (event == EVENT_RBUTTONDOWN) {
        // remove nearest manual point (if close)
        int nearest = -1; double minD = 1e9;
        for (int i = 0; i < manualPts.size(); i++) {
            double d = euclidDist(origPt, manualPts[i]);
            if (d < minD) { minD = d; nearest = i; }
        }
        if (nearest >= 0 && minD < 25.0 / scaleFactor) {
            manualPts.erase(manualPts.begin() + nearest);
        }
    }
}

// compute scaleFactor so image fits within MAX_WIN_W x MAX_WIN_H (and never >1) and produce scaled display image
void computeScaledDisplay() {
    double sx = (double)MAX_WIN_W / (double)imgOrig.cols;
    double sy = (double)MAX_WIN_H / (double)imgOrig.rows;
    scaleFactor = min({ 1.0, sx, sy });
    Size dstSz((int)round(imgOrig.cols * scaleFactor), (int)round(imgOrig.rows * scaleFactor));
    resize(imgOrig, imgDisplayScaled, dstSz);
}

int main(int argc, char** argv) {
    string path = (argc > 1) ? argv[1] : "resources/cards.jpg";
    imgOrig = imread(path);
    if (imgOrig.empty()) {
        cerr << "Cannot open " << path << "\n";
        return -1;
    }

    // optionally let user pass desired max window w/h as second/third args
    if (argc >= 3) MAX_WIN_W = atoi(argv[2]);
    if (argc >= 4) MAX_WIN_H = atoi(argv[3]);

    // compute scaled display (fits within MAX_WIN_W x MAX_WIN_H)
    computeScaledDisplay();

    // auto detect
    Mat pre = preProcessForContours(imgOrig);
    bool found = findDocumentContour(pre, autoPts);
    if (found) autoPts = reorderPoints(autoPts);

    // UI setup
    const string win = "DocScanner - fit-to-screen (press m to edit)";
    namedWindow(win, WINDOW_AUTOSIZE);
    setMouseCallback(win, onMouse, nullptr);

    Mat disp;
    Mat lastWarp;
    int saved = 0;

    cout << "Instructions:\n";
    cout << "  - window is scaled to fit " << MAX_WIN_W << "x" << MAX_WIN_H << " (change via args).\n";
    cout << "  - Press 'm' to toggle manual mode (add/drag/remove corners on scaled view).\n";
    cout << "  - 'a' copy auto->manual | 'r' reset manual | 'w' warp | 's' save warp | 'q' quit\n";

    while (true) {
        imgDisplayScaled.copyTo(disp);
        drawOverlay(disp);
        imshow(win, disp);
        int key = waitKey(10);
        if (key == -1) continue;
        if (key == 'm' || key == 'M') {
            manualMode = !manualMode;
            cout << (manualMode ? "Manual mode ON\n" : "Manual mode OFF\n");
        }
        else if (key == 'a' || key == 'A') {
            if (found) {
                manualPts.clear();
                for (auto& p : autoPts) manualPts.push_back(p); // store in orig coords
                cout << "Copied auto points to manual.\n";
            }
            else cout << "No auto-detected points to copy.\n";
        }
        else if (key == 'r' || key == 'R') {
            manualPts.clear();
            cout << "Manual points reset.\n";
        }
        else if (key == 'w' || key == 'W') {
            vector<Point2f> usePts;
            if (manualPts.size() == 4) usePts = manualPts;
            else if (found) usePts = autoPts;
            else {
                cout << "Need 4 manual points or auto-detected contour to warp.\n";
                continue;
            }
            auto ordered = reorderPoints(usePts);
            Mat warped = getWarpedA4(imgOrig, ordered); // IMPORTANT: pass original-image coords
            Mat warpedColor, warpedGray, finalBW;
            warpedColor = warped.clone(); // keep color version

            // Convert for scanned effect
            if (warped.channels() == 3) cvtColor(warped, warpedGray, COLOR_BGR2GRAY);
            else warpedGray = warped;

            Mat clahe;
            Ptr<CLAHE> c = createCLAHE(2.0, Size(8, 8));
            c->apply(warpedGray, clahe);
            adaptiveThreshold(clahe, finalBW, 255,
                ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 10);

            // store both results
            lastWarp = finalBW.clone();
            lastWarpColor = warpedColor.clone();

            imshow("Warped (B/W scanned)", finalBW);
            imshow("Warped (Color)", warpedColor);
            cout << "Warp applied. Press 's' to save B/W, 'c' to save Color.\n";
        }
        else if (key == 's' || key == 'S') {
            if (lastWarp.empty()) cout << "No B/W warped image to save. Press 'w' first.\n";
            else {
                string fn = "scanned_bw_" + to_string(saved++) + ".png";
                imwrite(fn, lastWarp);
                cout << "Saved " << fn << " (B/W)\n";
            }
        }
        else if (key == 'c' || key == 'C') {
            if (lastWarpColor.empty()) cout << "No color warped image to save. Press 'w' first.\n";
            else {
                string fn = "scanned_color_" + to_string(saved++) + ".png";
                imwrite(fn, lastWarpColor);
                cout << "Saved " << fn << " (Color)\n";
            }
        }
        else if (key == 'q' || key == 27) {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}
