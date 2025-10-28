#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "Core.hpp"
#include "imgui/ImGuiFileDialog.h"

using namespace cv;
using namespace std;
using namespace DocScanner;

struct AppState {
    Mat imgOrig, preview;
    Mat warpedBW, warpedColor;
    vector<Point2f> autoPts, manualPts;
    bool manualMode = false;
    bool foundAuto = false;
    int dragIdx = -1; // index of currently dragged point
    float scale = 1.0f;
    string filename = "";
} app;

// --- Helper: Convert cv::Mat -> OpenGL Texture
GLuint matToTexture(const Mat& mat) {
    if (mat.empty()) return 0;
    Mat rgb;
    if (mat.channels() == 1)
        cvtColor(mat, rgb, COLOR_GRAY2RGB);
    else
        cvtColor(mat, rgb, COLOR_BGR2RGB);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    return tex;
}

// --- Compute scaled preview
void computeScaledPreviewToFit(int maxW, int maxH) {
    if (app.imgOrig.empty()) {
        app.preview.release();
        app.scale = 1.0f;
        return;
    }
    if (maxW <= 0) maxW = 1;
    if (maxH <= 0) maxH = 1;
    double sx = (double)maxW / app.imgOrig.cols;
    double sy = (double)maxH / app.imgOrig.rows;
    app.scale = (float)min(sx, sy);
    Size dstSz((int)round(app.imgOrig.cols * app.scale), (int)round(app.imgOrig.rows * app.scale));
    if (dstSz.width <= 0) dstSz.width = 1;
    if (dstSz.height <= 0) dstSz.height = 1;
    resize(app.imgOrig, app.preview, dstSz);
}

// --- Load image and detect automatic document contour
bool loadImage(const string& path) {
    Mat img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Cannot open " << path << endl;
        return false;
    }
    app.imgOrig = img;

    Mat pre = preProcessForContours(app.imgOrig);
    app.foundAuto = findDocumentContour(pre, app.autoPts);
    if (app.foundAuto)
        app.autoPts = reorderPoints(app.autoPts);
    app.manualPts.clear();
    app.warpedBW.release();
    app.warpedColor.release();
    return true;
}

// --- Warp document using current points
void doWarp() {
    vector<Point2f> usePts;
    if (app.manualMode && app.manualPts.size() == 4)
        usePts = app.manualPts;
    else if (app.foundAuto)
        usePts = app.autoPts;
    else {
        cout << "Need 4 manual points or auto detection." << endl;
        return;
    }
    auto ordered = reorderPoints(usePts);
    Mat warped = getWarpedA4(app.imgOrig, ordered);
    app.warpedColor = warped.clone();
    app.warpedBW = makeBWScanEffect(warped);
}

// --- Main
int main(int argc, char** argv) {
    // start without preloaded image
    if (argc > 1) {
        // if user passed path on cmdline, try to load it
        app.filename = argv[1];
        if (!loadImage(app.filename)) {
            app.filename.clear();
        }
    }

    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "DocScanner (ImGui)", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    // Increase font size
    io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/arial.ttf", 24.0f);

    // Enlarge UI scale
    ImGui::GetStyle().ScaleAllSizes(2.0f);
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    GLuint texPreview = 0, texWarped = 0;
    string lastPreviewHash = "";
    string lastWarpedHash = "";
    int themeIndex = 0; // 0 = Light, 1 = Dark

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ===== Get window dimensions (put it here) =====
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        float controlWidth = display_w * 0.25f;
        float previewWidth = (display_w - controlWidth) * 0.5f;
        float warpedWidth = (display_w - controlWidth) * 0.5f;
        float fullHeight = display_h;

        // ==== Controls ====
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(controlWidth, fullHeight));
        ImGui::Begin("Controls", nullptr,
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse);

        ImGui::Text("File: %s", app.filename.empty() ? "<none>" : app.filename.c_str());

        if (ImGui::Button("Load Image...")) {
            IGFD::FileDialogConfig config;
            config.path = ".";
            config.countSelectionMax = 1;
            config.flags = ImGuiFileDialogFlags_Modal;

            // set minimum initial window size (user can still resize)
            ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_Appearing);

            // Open the dialog with filters and configuration
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseImageDlgKey",
                "Choose Image",
                "Image files{.png,.jpg,.jpeg,.bmp,.tif,.tiff}",
                config
            );
        }

        // Display the dialog if it’s open
        if (ImGuiFileDialog::Instance()->Display("ChooseImageDlgKey")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string filePath = ImGuiFileDialog::Instance()->GetFilePathName();
                app.filename = filePath;
                if (!loadImage(app.filename)) {
                    // failed -> clear
                    app.filename.clear();
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }
        ImGui::SameLine();
        if (ImGui::Button("Warp")) doWarp();
        if (ImGui::Button("Save BW") && !app.warpedBW.empty()) {
            IGFD::FileDialogConfig config;
            config.path = ".";
            config.countSelectionMax = 1;
            config.flags = ImGuiFileDialogFlags_Modal;

            // set minimum initial window size (user can still resize)
            ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_Appearing);

            ImGuiFileDialog::Instance()->OpenDialog(
                "SaveBWDialog",
                "Save BW Image As...",
                "PNG files{.png},JPEG files{.jpg,.jpeg},Bitmap files{.bmp},All files{.*}",
                config
            );
        }

        if (ImGui::Button("Save Color") && !app.warpedColor.empty()) {
            IGFD::FileDialogConfig config;
            config.path = ".";
            config.countSelectionMax = 1;
            config.flags = ImGuiFileDialogFlags_Modal;

            // set minimum initial window size (user can still resize)
            ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_Appearing);

            ImGuiFileDialog::Instance()->OpenDialog(
                "SaveColorDialog",
                "Save Color Image As...",
                "PNG files{.png},JPEG files{.jpg,.jpeg},Bitmap files{.bmp},All files{.*}",
                config
            );
        }
        // --- Save dialogs handling ---
        if (ImGuiFileDialog::Instance()->Display("SaveBWDialog")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string savePath = ImGuiFileDialog::Instance()->GetFilePathName();
                try {
                    imwrite(savePath, app.warpedBW);
                    cout << "Saved BW image to: " << savePath << endl;
                }
                catch (const cv::Exception& e) {
                    cerr << "Failed to save BW image: " << e.what() << endl;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("SaveColorDialog")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string savePath = ImGuiFileDialog::Instance()->GetFilePathName();
                try {
                    imwrite(savePath, app.warpedColor);
                    cout << "Saved color image to: " << savePath << endl;
                }
                catch (const cv::Exception& e) {
                    cerr << "Failed to save color image: " << e.what() << endl;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::Separator();
        ImGui::Text("Theme");

        const char* themes[] = { "Light", "Dark" };
        if (ImGui::Combo("##ThemeCombo", &themeIndex, themes, IM_ARRAYSIZE(themes))) {
            if (themeIndex == 0)
                ImGui::StyleColorsLight();
            else
                ImGui::StyleColorsDark();
        }
        ImGui::Separator();

        bool oldMode = app.manualMode;
        ImGui::Checkbox("Manual Mode", &app.manualMode);
        if (app.manualMode && !oldMode) {
            // switched from auto → manual
            if (app.foundAuto)
                app.manualPts = app.autoPts;
        }
        else if (!app.manualMode && oldMode) {
            // switched from manual → auto
            app.manualPts.clear();
        }

        ImGui::End();

        // ==== Image View ====
        ImGui::SetNextWindowPos(ImVec2(controlWidth, 0));
        ImGui::SetNextWindowSize(ImVec2(previewWidth, fullHeight));
        ImGui::Begin("Image View", nullptr,
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse);

        // Determine available size inside Image View to fit the image
        ImVec2 avail = ImGui::GetContentRegionAvail();
        int availW = (int)avail.x;
        int availH = (int)avail.y;

        // compute preview sized to fit avail area
        computeScaledPreviewToFit(availW, availH);

        ImVec2 imgStart = ImGui::GetCursorScreenPos();
        ImVec2 imgSize((float)app.preview.cols, (float)app.preview.rows);

        // draw placeholder if no image
        if (app.preview.empty()) {
            ImGui::Dummy(ImVec2((float)min(availW, 400), (float)min(availH, 300)));
            ImGui::SameLine();
            ImGui::TextWrapped("No image loaded.\nClick 'Load Image...' to open an image.");
        }
        else {
            // ensure texture updated for preview
            if (texPreview) { glDeleteTextures(1, &texPreview); texPreview = 0; }
            texPreview = matToTexture(app.preview);

            // center the image inside the available region horizontally/vertically
            ImVec2 curCursor = ImGui::GetCursorScreenPos();
            float padX = (avail.x - imgSize.x) * 0.5f;
            float padY = (avail.y - imgSize.y) * 0.5f;
            if (padX < 0) padX = 0;
            if (padY < 0) padY = 0;
            ImGui::SetCursorScreenPos(ImVec2(curCursor.x + padX, curCursor.y + padY));
            ImVec2 drawStart = ImGui::GetCursorScreenPos();

            ImGui::Image((ImTextureID)(intptr_t)texPreview, imgSize);

            // Submit a dummy to inform ImGui of the item’s size (prevents Dear ImGui warning)
            ImGui::Dummy(ImVec2(avail.x, avail.y));

            // prepare draw list for overlays
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 itemMin = drawStart;
            ImVec2 mousePos = io.MousePos;

            // --- Dragging behavior (only when mouse inside image)
            if (app.manualMode && !app.manualPts.empty()) {
                ImVec2 imgMin = itemMin;
                ImVec2 imgMax(imgMin.x + imgSize.x, imgMin.y + imgSize.y);
                bool mouseInImage = ImGui::IsMouseHoveringRect(imgMin, imgMax);

                if (mouseInImage && ImGui::IsMouseClicked(0)) {
                    for (int i = 0; i < 4; ++i) {
                        ImVec2 p(itemMin.x + app.manualPts[i].x * app.scale,
                            itemMin.y + app.manualPts[i].y * app.scale);
                        float dx = mousePos.x - p.x;
                        float dy = mousePos.y - p.y;
                        if ((dx * dx + dy * dy) < 100.0f) {
                            app.dragIdx = i;
                            break;
                        }
                    }
                }

                if (mouseInImage && ImGui::IsMouseDown(0) && app.dragIdx >= 0) {
                    float x = (mousePos.x - itemMin.x) / app.scale;
                    float y = (mousePos.y - itemMin.y) / app.scale;
                    // clamp to image bounds (original image coordinates)
                    x = std::max(0.0f, std::min(x, (float)app.imgOrig.cols));
                    y = std::max(0.0f, std::min(y, (float)app.imgOrig.rows));
                    app.manualPts[app.dragIdx] = Point2f(x, y);
                }

                if (ImGui::IsMouseReleased(0))
                    app.dragIdx = -1;
            }

            // --- Draw polygons (use draw_list so overlay is on top)
            auto drawPoly = [&](const vector<Point2f>& pts, ImU32 color) {
                if (pts.size() != 4) return;
                for (int i = 0; i < 4; ++i) {
                    ImVec2 a(itemMin.x + pts[i].x * app.scale, itemMin.y + pts[i].y * app.scale);
                    ImVec2 b(itemMin.x + pts[(i + 1) % 4].x * app.scale, itemMin.y + pts[(i + 1) % 4].y * app.scale);
                    draw_list->AddLine(a, b, color, 2.0f);
                    draw_list->AddCircleFilled(a, 6.0f, color);
                }
                };

            if (app.foundAuto && !app.manualMode)
                drawPoly(app.autoPts, IM_COL32(0, 150, 255, 255)); // blue auto
            if (app.manualMode && !app.manualPts.empty())
                drawPoly(app.manualPts, IM_COL32(0, 255, 100, 255)); // green manual
        }

        ImGui::End();

        // --- Warped preview window ---
        ImGui::SetNextWindowPos(ImVec2(controlWidth + previewWidth, 0));
        ImGui::SetNextWindowSize(ImVec2(warpedWidth, fullHeight));
        ImGui::Begin("Warped Preview", nullptr,
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse);

        ImVec2 availWarp = ImGui::GetContentRegionAvail();
        // show placeholder when no warped image
        if (app.warpedColor.empty()) {
            ImGui::Dummy(ImVec2((float)min((int)availWarp.x, 400), (float)min((int)availWarp.y, 300)));
            ImGui::SameLine();
            ImGui::TextWrapped("No warped image.\nClick 'Warp' to get a scanned preview.");
        }
        else {
            // compute warped preview size to fit availWarp while preserving aspect ratio
            int maxW = (int)availWarp.x;
            int maxH = (int)availWarp.y;
            double sx = (double)maxW / app.warpedColor.cols;
            double sy = (double)maxH / app.warpedColor.rows;
            double s = min(1.0, min(sx, sy));
            ImVec2 warpedSize((float)(app.warpedColor.cols * s), (float)(app.warpedColor.rows * s));

            // update warped texture
            if (texWarped) { glDeleteTextures(1, &texWarped); texWarped = 0; }
            texWarped = matToTexture(app.warpedColor);

            // center
            ImVec2 cur = ImGui::GetCursorScreenPos();
            float padX = (availWarp.x - warpedSize.x) * 0.5f; if (padX < 0) padX = 0;
            float padY = (availWarp.y - warpedSize.y) * 0.5f; if (padY < 0) padY = 0;
            ImGui::SetCursorScreenPos(ImVec2(cur.x + padX, cur.y + padY));
            ImGui::Image((ImTextureID)(intptr_t)texWarped, warpedSize);

            // Submit dummy to mark used area (prevents Dear ImGui layout warning)
            ImGui::Dummy(ImVec2(availWarp.x, availWarp.y));
        }
        ImGui::End();

        // --- Render ---
        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // cleanup textures
    if (texPreview) glDeleteTextures(1, &texPreview);
    if (texWarped) glDeleteTextures(1, &texWarped);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}