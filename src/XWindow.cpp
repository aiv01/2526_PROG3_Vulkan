#include "XWindow.h"
#include "Common.h"

XWindow::XWindow(int InWidth, int InHeight, const std::string& InTitle)
    : Width(InWidth), Height(InHeight), Title(InTitle), DeltaTime(0.f)
{
    CHECK_DIE(glfwInit(), "GLFW Initialization");

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No OpenGL context

	GLFWmonitor* Monitor = NULL; //windowed mode
	RawWindow = glfwCreateWindow(Width, Height, Title.c_str(), Monitor, NULL);
    //my add MONITOR CHECK
    int X, Y;
    glfwGetWindowPos(RawWindow, &X, &Y);//should get the raw position
    int Count;
    GLFWmonitor** Monitors = glfwGetMonitors(&Count);//found this function TODO check if works

    for (int i = 0; i < Count; ++i)//check for all monitors and save the one where born the window
    {
        int MX, MY;
        glfwGetMonitorPos(Monitors[i], &MX, &MY);
        const GLFWvidmode* Mode = glfwGetVideoMode(Monitors[i]);

        if (X >= MX && X < MX + Mode->width && Y >= MY && Y < MY + Mode->height)
        {
            CurrentMonitor = Monitors[i];
            break;
        }
    }//my add end


    CHECK_DIE(RawWindow, "GLFW Create Window");

    glfwSetWindowUserPointer(RawWindow, this);
    glfwSetFramebufferSizeCallback(RawWindow, [](GLFWwindow* w, int, int) {
        static_cast<XWindow*>(glfwGetWindowUserPointer(w))->Context.NotifyFramebufferResized();
    });
    //my add monitor movement/change callback
    glfwSetWindowPosCallback(
    RawWindow,
    [](GLFWwindow* W, int, int)
    {
        auto* Window =
            static_cast<XWindow*>(glfwGetWindowUserPointer(W));

        Window->CheckMonitorChange();
    });

    Context.Init(RawWindow);
}

//my add change monitor
void XWindow::CheckMonitorChange()
{
    int X, Y;
    glfwGetWindowPos(RawWindow, &X, &Y);//same raw window position and all monitor screening
    int Count;
    GLFWmonitor** Monitors = glfwGetMonitors(&Count);
    GLFWmonitor* NewMonitor = nullptr;

    for (int i = 0; i < Count; ++i)//check for new actual monitor using position of the window 
    {
        int MX, MY;
        glfwGetMonitorPos(Monitors[i], &MX, &MY);

        const GLFWvidmode* Mode =glfwGetVideoMode(Monitors[i]);//source help Stack Overflow

        if (X >= MX && X < MX + Mode->width && Y >= MY && Y < MY + Mode->height)//here check if we are over that monitor
        {
            NewMonitor = Monitors[i];
            break;
        }
    }
    //TODO add a controll: change the monitor if different
    if (NewMonitor != CurrentMonitor)
    {
        CurrentMonitor = NewMonitor;
        LOG_DEBUG("Monitor changed");
        Context.NotifyFramebufferResized();//adapt to new monitor
    }
    //TODO test this part
    //TO DO windowss size change resistance
}

XWindow::~XWindow()
{
    Context.WaitIdle();
    Context.Cleanup();
    glfwDestroyWindow(RawWindow);
    glfwTerminate();
}

void XWindow::Update() 
{
    static float LastTime = glfwGetTime();
    float CurrentTime = glfwGetTime();
    DeltaTime = CurrentTime - LastTime;
    LastTime = CurrentTime;

	glfwPollEvents();
    Context.DrawFrame();
}

bool XWindow::IsOpened() const 
{
    return !glfwWindowShouldClose(RawWindow);
}

void XWindow::SetTitle(const std::string& InTitle)
{
    Title = InTitle;
    glfwSetWindowTitle(RawWindow, Title.c_str());
}

float XWindow::GetDeltaTime() const
{
    return DeltaTime;
}

void XWindow::SetVSync(bool InEnabled)
{
    // Note: This approach is no-more valid
    /*
    Control VSync: 0 = disable, 1 = enabled (default).
    int Value = InEnabled ? 1 : 0;
    glfwSwapInterval(Value);
    */
   Context.SetVSync(InEnabled);
}

int XWindow::GetWidth() const
{
    return Width;
}

int XWindow::GetHeight() const
{
    return Height;
}
