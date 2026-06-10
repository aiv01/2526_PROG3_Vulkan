#pragma once
#include <string>
#include "VulkanContext.h"

class XWindow
{
public:
    XWindow(int InWidth, int InHeight, const std::string& InTitle);
    ~XWindow();
    void Update();
    bool IsOpened() const;
    void SetTitle(const std::string& InTitle);
    float GetDeltaTime() const;
    void SetVSync(bool InEnabled);
    int GetWidth() const;
    int GetHeight() const;
    void CheckMonitorChange();

private:
    int Width;
    int Height;
    std::string Title;
    GLFWwindow* RawWindow;
    float DeltaTime;
    VulkanContext Context;
    GLFWmonitor* CurrentMonitor = nullptr; //my add
};