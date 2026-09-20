import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import place_poles
from scipy.linalg import solve_continuous_are

xml_path = 'model.xml' #xml file (assumes this is in the same folder as this file)
simend = 100 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass


ref = {"x": 0.0}
STEP, XMAX = 0.1, 2.0   


outc=[]

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    m1, l, g = 1.0, 1.0, 9.81 
    A = np.array([[0,1,0,0],[0,0,-0.7178,0],[0,0,0,1],[0,0,15.972,0]])
    B = np.array([[0],[0.9756],[0],[-1.4634]])
    Q = np.diag([10,1,10,1])
    R = np.array([[0.5]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    X = np.array([data.qpos[0],data.qvel[0],data.qpos[1], data.qvel[1]])
    X_ref = np.array([ref['x'],0,0,0])
    out = np.clip(-K@(X.T - X_ref.T), -10, 10)
    outc.append(out)
    data.ctrl = out


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
def keyboard(window, key, scancode, act, mods):
    if act == glfw.RELEASE:
        return
    if key == glfw.KEY_RIGHT:
        ref["x"] = min(ref["x"] + STEP, XMAX)
    elif key == glfw.KEY_LEFT:
        ref["x"] = max(ref["x"] - STEP, -XMAX)
        mj.mj_forward(model, data)
    glfw.set_window_title(window, f"x_ref = {ref['x']:.2f}")


# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = 90.0 ; cam.elevation = -12.399999999999995 ; cam.distance =  5.301661093309673
cam.lookat =np.array([ 0.029271066813332226 , -0.0002043539252718035 , 0.5611881268426189 ])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

data.qpos[0]=0
data.qpos[1]=np.deg2rad(0)

theta=[]
x=[]

t_push, dur, F = 2.0, 0.05, 10.0

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        data.xfrc_applied[1, 0] = F if t_push <= data.time < t_push + dur else 0.0
        mj.mj_step(model, data)

    theta.append(data.qpos[1])
    x.append(data.qpos[0])
    
    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
t=np.linspace(0,data.time,len(theta))
tc=np.linspace(0,data.time,len(outc))

def settling_time(t, sig, tol, t_start=0.0):
    outside = np.abs(sig) > tol
    if not outside.any():
        return 0.0
    last = np.where(outside)[0][-1]
    if last == len(t) - 1:
        return np.inf              
    return t[last + 1] - t_start

xt = settling_time(t,x,0.1)
thetat = settling_time(t,theta,0.1)

print("x Settling time = ", xt)
print("theta Settling time = ", thetat)
print("Peak Force = ", np.max(outc))
sum=0

for i in range(len(outc)):
    sum = sum + (outc[i]**2)*data.time/len(outc)

print("Total Control Effort = ", sum)

plt.figure()
plt.subplot(3,1,1)
plt.plot(t, theta)
plt.ylabel('theta (rad)')
plt.subplot(3,1,2)
plt.plot(t, x)
plt.ylabel('x (m)')
plt.subplot(3,1,3)
plt.plot(tc, outc)
plt.ylabel('control output')
plt.xlabel('time (s)')

plt.show()