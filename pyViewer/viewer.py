import os
import numpy as np
import pyViewer.transformations as tf
import moderngl as mgl
import PIL
from PIL import Image
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, FULLSCREEN
from pyglfw import pyglfw
from OpenGL.GL import glViewport, glEnable, GL_VERTEX_PROGRAM_POINT_SIZE, glBegin, glEnd, GL_LINES, glVertex3fv, \
    glColor3fv, glColor3f, glLineWidth, glDisable, GL_LIGHTING, glLoadIdentity, glUseProgram, glLoadMatrixf, GL_DEPTH_TEST, \
    glMatrixMode, GL_PROJECTION, GL_MODELVIEW, GL_BLEND, glWindowPos2dv, glIsEnabled, glReadPixels, GL_FLOAT, GL_DEPTH_COMPONENT

from OpenGL.GLUT import glutBitmapCharacter, glutInit
from OpenGL.GLUT.fonts import GLUT_BITMAP_9_BY_15
# from OpenGL.GLU import gluOrtho2D

'''
REQUIREMENTS

pip install ModernGL numpy PIL pywavefront
pip install pyglfw              # Only for glfw window management (recommended)
pip install pygame              # Only for pygame window management
pip install pybullet            # Only needed for pybullet integration

TODO:
- Shadows
- Fix lighting?
- Text on window
- Floating text
- Primitive geometry makers
- Remove transformations.py dependency

- GUI
    - Screenshot button
    - Focus on cursor point after pressing F        
'''

point_cloud_vertex_shader = ''' 
#version 330 core

uniform mat4 Mvp;
uniform int psize;

in vec3 in_vert;
in vec4 in_color;

out vec4 v_color;

void main() {
	v_color = in_color;
	gl_Position = Mvp * vec4(in_vert, 1.0);
	gl_PointSize = psize;
}
'''

point_cloud_fragment_shader = ''' 
#version 330 core

in vec4 v_color;

out vec4 f_color;

void main()
{
    f_color = v_color;
}
'''

geometry_vertex_shader = '''
#version 330 core

uniform mat4 Mvp;

in vec3 in_vert;
in vec3 in_norm;
in vec3 in_text;
in vec4 in_color;

out vec3 v_vert;
out vec3 v_norm;
out vec3 v_text;
out vec4 v_color;

void main() {
	v_vert = in_vert;
	v_norm = in_norm;
	v_text = in_text;
	v_color = in_color;
	gl_Position = Mvp * vec4(v_vert, 1.0);
} 
'''

geometry_fragment_shader = '''
#version 330 core

uniform sampler2D Texture;
uniform vec3 Light;

in vec3 v_vert;
in vec3 v_norm;
in vec3 v_text;
in vec4 v_color;

out vec4 f_color;

void main()
{
    float lum = dot(normalize(v_norm), normalize(v_vert - Light));
    lum = acos(lum) / 3.14159265;
    lum = clamp(lum, 0.0, 0.1);

    vec4 color = texture(Texture, v_text.xy);

    f_color = vec4(color.rgb + v_color.rgb * v_color.a * lum, color.a);
}
'''

image_vertex_shader = '''
#version 330 core
layout (location=0) in vec2 in_vert;
layout (location=1) in vec2 in_text;
layout (location=2) in vec4 in_color;

out vec4 ourColor;
out vec2 TexCoord;

void main()
{
    gl_Position = vec4(in_vert,0.0f,1.0f);
    ourColor = in_color;
    TexCoord= vec2(in_text.x,in_text.y);
}
'''

image_fragment_shader = '''
#version 330 core
in vec4 ourColor;
in vec2 TexCoord;

out vec4 f_color;

uniform sampler2D Texture;

void main()
{
    f_color = texture(Texture, TexCoord) + ourColor;
}
'''


class CTransform(object):
    def __init__(self, t=None):
        self.t = np.eye(4)
        if t is not None:
            self.t = t

    def __mul__(self, other):
        return CTransform(np.matmul(self.t, other.t))

    def __matmul__(self, other):
        return CTransform(np.matmul(self.t, other.t))

    def __repr__(self):
        return " pos: " + repr(tuple(self.t[0:3, 3])) + " rot: " + repr(tf.euler_from_matrix(self.t))

    def look_at(self, focus=(0, 0, 0), up=(0, 0, 1)):
        z_vec = focus - self.t[0:3, 3]
        x_vec = np.cross(up, z_vec)
        y_vec = np.cross(z_vec, x_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)
        z_vec = z_vec / np.linalg.norm(z_vec)
        translation = self.t[0:3, 3]
        matrix = np.hstack((x_vec.reshape(3, 1), y_vec.reshape(3, 1), z_vec.reshape(3, 1), translation.reshape(3, 1)))
        matrix = np.vstack((matrix, np.array([0, 0, 0, 1])))
        return matrix


class CCamera(object):
    def __init__(self, alpha=0.7, beta=0.7, distance=2.0, focus=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
        self.r = distance
        self.alpha = alpha
        self.beta = beta
        self.focus_point = np.array(focus)
        self.up_vector = np.array(up)
        self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
        self.sensitivity = 0.02

    #TODO: Remove pygame specific keys here
    def process_event(self, event):
        if event.type == CEvent.KEYDOWN:
            if event.data[0] == pygame.K_w:
                self.focus_point = self.focus_point + np.array([0, 0, -0.1])
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
            if event.data[0] == pygame.K_a:
                self.focus_point = self.focus_point + np.array([-0.1, 0.0, 0.0])
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
            if event.data[0] == pygame.K_s:
                self.focus_point = self.focus_point + np.array([0, 0, 0.1])
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
            if event.data[0] == pygame.K_d:
                self.focus_point = self.focus_point + np.array([0.1, 0.0, 0.0])
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
            if event.data[0] == pygame.K_q:
                self.focus_point = self.focus_point + np.array([0, 0.1, 0.0])
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
            if event.data[0] == pygame.K_e:
                self.focus_point = self.focus_point + np.array([0, -0.1, 0.0])
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)

            if event.data[0] == pygame.K_c:
                self.alpha = 0.0
                self.beta = 0.0
                self.r = 5.0
                self.focus_point = np.array([0.0, 0.0, 0.0])
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
                print("RESET CAMERA")
                print("Camera: a:", self.alpha, " b:", self.beta, " r:", self.r)

        if event.type == CEvent.MOUSEBUTTONDOWN:
            if event.data[1] == 4:
                self.r = self.r - 0.1
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
                print("Camera: a:", self.alpha, " b:", self.beta, " r:", self.r)
            if event.data[1] == 5:
                self.r = self.r + 0.1
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
                print("Camera: a:", self.alpha, " b:", self.beta, " r:", self.r)

        if event.type == CEvent.MOUSEMOTION:
            if event.data[2][0] and np.abs(event.data[1][0]) < 50 and np.abs(event.data[1][1]) < 50:
                self.alpha = self.alpha + event.data[1][0] * self.sensitivity
                if self.alpha > 2*np.pi or self.alpha < -2*np.pi:
                    self.alpha = 0

                self.beta = self.beta + event.data[1][1] * self.sensitivity
                if self.beta > 2*np.pi or self.beta < -2*np.pi:
                    self.beta = 0

                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
                print("Camera: a:", self.alpha, " b:", self.beta, " r:", self.r)

            if event.data[2][1] and np.abs(event.data[1][0]) < 50 and np.abs(event.data[1][1]) < 50:
                self.focus_point = self.focus_point + event.data[1][0] * self.camera_matrix[0, 0:3] * self.sensitivity
                self.focus_point = self.focus_point + event.data[1][1] * self.camera_matrix[1, 0:3] * self.sensitivity
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
                print("Camera: focus:", self.focus_point)

            if event.data[2][2]:
                pass

    def look_at(self, focus=(0, 0, 0), up=(0, 0, 1)):

        position = np.array([0.0, 0.0, 0.0])
        alpha = self.alpha
        if alpha > np.pi:
            alpha = -np.pi + (alpha - np.pi)
        elif alpha < -np.pi:
            alpha = np.pi + (alpha + np.pi)

        beta = self.beta
        if beta > np.pi/2:
            beta = np.pi/2 - 0.01
        elif beta < -np.pi/2:
            beta = -np.pi/2 + 0.01
        position[0] = self.r * np.cos(alpha) * np.cos(beta)
        position[1] = self.r * np.sin(alpha) * np.cos(beta)
        position[2] = self.r * np.sin(beta)

        # print("Dist: ", np.sqrt(np.matmul(position, position.transpose())))

        # print("Camera pos: ", position)

        z_vec = position - focus
        z_vec = z_vec / np.linalg.norm(z_vec)
        x_vec = np.cross(up, z_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)
        y_vec = np.cross(z_vec, x_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)

        trans = tf.compose_matrix(translate=-position)
        rot = np.eye(4)
        rot[0, 0:3] = x_vec
        rot[1, 0:3] = y_vec
        rot[2, 0:3] = z_vec
        # rot[0:3, 0] = x_vec
        # rot[0:3, 1] = y_vec
        # rot[0:3, 2] = z_vec

        matrix = np.matmul(rot, trans)

        return matrix


class CEvent(object):
    QUIT = 0
    KEYDOWN = 1             # unicode, key, mod
    KEYUP = 2               # key, mod
    MOUSEMOTION = 3         # pos, rel, buttons
    MOUSEBUTTONUP = 4       # pos, button
    MOUSEBUTTONDOWN = 5     # pos, button (0-left, 1-mid, 2-left, 4-scrollup, 5-scrolldown)
    VIDEORESIZE = 6         # size, w, h
    VIDEOEXPOSE = 7

    def __init__(self):
        self.type = None
        self.data = None
        self.source_obj = None

    def initialize(self, event):
        self.source_obj = event


############################################################################################################
# WINDOW MANAGER IMPLEMENTATIONS
############################################################################################################
class CWindowManager(object):
    @staticmethod
    def init_display():
        raise NotImplementedError()

    @staticmethod
    def set_window_name(name):
        raise NotImplementedError()

    @staticmethod
    def get_events():
        raise NotImplementedError()

    @staticmethod
    def set_window_mode(size, options):
        raise NotImplementedError()

    @staticmethod
    def set_mouse_pos(x, y):
        raise NotImplementedError()

    @staticmethod
    def get_mouse_pos():
        raise NotImplementedError()


class COffscreenWindowManager(CWindowManager):
    @staticmethod
    def init_display():
        pass

    @staticmethod
    def set_window_name(name):
        pass

    @staticmethod
    def get_events():
        return []

    @staticmethod
    def set_window_mode(size, options):
        pass

    @staticmethod
    def set_mouse_pos(x, y):
        pass

    @staticmethod
    def get_mouse_pos():
        return [0, 0]

    def draw(self):
        pass


class CPygameEvent(CEvent):
    def __init__(self):
        super(CPygameEvent, self).__init__()

    def initialize(self, event):
        self.source_obj = event
        if event.type == pygame.QUIT:
            self.type = CEvent.QUIT
        elif event.type == pygame.KEYDOWN:
            self.type = CEvent.KEYDOWN
            self.data = (event.key, event.mod, event.unicode)
        elif event.type == pygame.KEYUP:
            self.type = CEvent.KEYUP
            self.data = (event.key, event.mod)
        elif event.type == pygame.MOUSEMOTION:
            self.type = CEvent.MOUSEMOTION
            self.data = (event.pos, event.rel, event.buttons)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.type = CEvent.MOUSEBUTTONUP
            self.data = (event.pos, event.button)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.type = CEvent.MOUSEBUTTONDOWN
            self.data = (event.pos, event.button)
        elif event.type == pygame.VIDEORESIZE:
            self.type = CEvent.VIDEORESIZE
            self.data = (event.size, event.w, event.h)
        elif event.type == pygame.VIDEOEXPOSE:
            self.type = CEvent.VIDEOEXPOSE


class CPygameWindowManager(CWindowManager):
    @staticmethod
    def init_display():
        pygame.init()
        pygame.font.init()

    @staticmethod
    def set_window_name(name):
        pygame.display.set_caption(name)

    @staticmethod
    def get_events():
        events = pygame.event.get()
        event_list = []
        for ev in events:
            event = CPygameEvent()
            event.initialize(ev)
            event_list.append(event)
        return event_list

    @staticmethod
    def draw():
        pygame.display.flip()
        pygame.time.wait(1)

    @staticmethod
    def set_window_mode(size, options):
        return pygame.display.set_mode(size, options)

    @staticmethod
    def set_mouse_pos(x, y):
        pygame.mouse.set_pos(x, y)

    @staticmethod
    def get_mouse_pos():
        pos = pygame.mouse.get_pos()
        return pos


class CGLFWWindowManager(CWindowManager):
    def init_display(self):
        if not pyglfw.init():
            os.sys.exit(1)

        pyglfw.Window.hint(visible=False)
        self.window = pyglfw.Window(200, 200, "")
        self.window.make_current()
        self.window.show()
        self.window.set_key_callback(CGLFWWindowManager.key_callback)
        self.window.set_char_callback(CGLFWWindowManager.char_callback)
        self.window.set_scroll_callback(CGLFWWindowManager.scroll_callback)
        self.window.set_mouse_button_callback(CGLFWWindowManager.mouse_button_callback)
        self.window.set_cursor_enter_callback(CGLFWWindowManager.cursor_enter_callback)
        self.window.set_cursor_pos_callback(CGLFWWindowManager.cursor_pos_callback)
        self.window.set_window_size_callback(CGLFWWindowManager.window_size_callback)
        self.window.set_window_pos_callback(CGLFWWindowManager.window_pos_callback)
        self.window.set_window_close_callback(CGLFWWindowManager.window_close_callback)
        self.window.set_window_refresh_callback(CGLFWWindowManager.window_refresh_callback)
        self.window.set_window_focus_callback(CGLFWWindowManager.window_focus_callback)
        self.window.set_window_iconify_callback(CGLFWWindowManager.window_iconify_callback)
        self.window.set_framebuffer_size_callback(CGLFWWindowManager.framebuffer_size_callback)
        self.window.event_queue = []
        self.window.cur_posx = self.window.cursor_pos[0]
        self.window.cur_posy = self.window.cursor_pos[1]

    def key_callback(self, key, scancode, action, mods):
        ev = CEvent()
        if action == pyglfw.api.GLFW_RELEASE:
            ev.type = CEvent.KEYUP
        if action == pyglfw.api.GLFW_PRESS:
            ev.type = CEvent.KEYDOWN

        ev.data = (key, scancode, mods)
        self.event_queue.append(ev)
        print("keybrd: key=%s scancode=%s action=%s mods=%s" % (key, scancode, action, mods))

    def char_callback(self, char):
        pass
        # print("unichr: char=%s" % char)

    def scroll_callback(self, off_x, off_y):
        ev = CEvent()
        ev.type = CEvent.MOUSEBUTTONDOWN
        if off_y > 0:
            ev.data = (self.cursor_pos, 4)
        elif off_y < 0:
            ev.data = (self.cursor_pos, 5)
        self.event_queue.append(ev)
        # print("scroll: x=%s y=%s" % (off_x, off_y))

    def mouse_button_callback(self, button, action, mods):
        # print("button: button=%s action=%s mods=%s" % (button, action, mods))
        pass

    def cursor_enter_callback(self, status):
        pass
        # print("cursor: status=%s" % status)

    def cursor_pos_callback(self, pos_x, pos_y):
        ev = CEvent()
        ev.type = CEvent.MOUSEMOTION
        buttons = (self.mice.left, self.mice.middle, self.mice.right)
        rel_x = pos_x - self.cur_posx
        rel_y = pos_y - self.cur_posy
        ev.data = ((pos_x, pos_y), (rel_x, rel_y), buttons)
        self.event_queue.append(ev)
        self.cur_posx = pos_x
        self.cur_posy = pos_y
        # print("curpos: x=%s y=%s" % (pos_x, pos_y))

    def window_size_callback(self, wsz_w, wsz_h):
        ev = CEvent()
        ev.type = CEvent.VIDEORESIZE
        ev.data = ((wsz_w, wsz_h), wsz_w, wsz_h)
        found = False
        for e in self.event_queue:
            if e.type == CEvent.VIDEORESIZE:
                e.data = ev.data
                found = True
        if not found:
            self.event_queue.append(ev)
        # print("window: w=%s h=%s" % (wsz_w, wsz_h))

    def window_pos_callback(self, pos_x, pos_y):
        pass
        # print("window: x=%s y=%s" % (pos_x, pos_y))

    def window_close_callback(self):
        ev = CEvent()
        ev.type = CEvent.QUIT
        self.event_queue.append(ev)

    def window_refresh_callback(self):
        pass
        # print("redraw")

    def window_focus_callback(self, status):
        pass
        # print("active: status=%s" % status)

    def window_iconify_callback(self, status):
        pass
        # print("hidden: status=%s" % status)

    def framebuffer_size_callback(self, fbs_x, fbs_y):
        pass
        # print("buffer: x=%s y=%s" % (fbs_x, fbs_y))

    def set_window_name(self, name):
        if isinstance(name, str):
            self.window.set_title(name.encode("utf8"))
        elif isinstance(name, bytes):
            self.window.set_title(name)
        else:
            self.window.set_title(str(name).encode("utf8"))

    def get_events(self):
        self.window.event_queue.clear()
        pyglfw.poll_events()
        return self.window.event_queue

    def draw(self):
        self.window.swap_buffers()

    def set_window_mode(self, size, options=None):
        self.window.size = size

    def set_mouse_pos(self, x, y):
        self.window.cursor_pos = (x, y)

    def get_mouse_pos(self):
        pos = self.window.cursor_pos
        return pos
############################################################################################################
# END OF WINDOW MANAGER IMPLEMENTATIONS
############################################################################################################


class CScene(object):
    def __init__(self, name="PyViewer", width=800, height=600, location=(0, 0), window_manager=CGLFWWindowManager(), near=0.001, far = 100.0):
        print("ModernGL: ", mgl.__version__)
        self.ctx = mgl.create_standalone_context()

        self.width = width
        self.height = height

        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))
        self.wm = window_manager
        self.wm.fbo = self.fbo
        self.fbo.use()

        self.init_display(name, width, height, location)

        self.ctx.viewport = (0, 0, self.width, self.height)
        self.root = CNode(id=0, parent=None, transform=CTransform(), geometry=None, material=None)
        self.nodes = [self.root]

        self.camera = CCamera()
        self.render_mode = mgl.TRIANGLES

        aspect = self.width / float(self.height)
        self.near = near
        self.far = far
        self.depth_modes_nonlinear = 0
        self.depth_modes_linear = 1
        self.depth_mode = self.depth_modes_nonlinear
        self.FoV = 60.0
        self.perspective = self.compute_perspective_matrix(self.FoV, self.FoV / aspect, near, far)

        self.glut_init = False

    # def __del__(self):
    #     self.fbo.release()
    #     self.ctx.release()

    #############################################################
    # GENERIC INITIALIZATION METHODS. WRAPPER TO HANDLE MULTIPLE WINDOW MANAGERS (pygame, pyglfw, ...)
    # current window manager = glfw
    #############################################################
    def init_display(self, name, width, height, location, options=None):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % location  # TODO: Verify this is not platform specific
        self.wm.init_display()

        self.width = width
        self.height = height
        self.wm.set_window_mode((self.width, self.height), options=options)

        self.wm.set_window_name(name)

    def swap_buffers(self):
        self.wm.draw()

    def get_events(self):
        return self.wm.get_events()

    def set_window_name(self, name):
        self.wm.set_window_name(name)

    def set_window_mode(self, size, options):
        return self.wm.set_window_mode(size, options)

    @staticmethod
    def compute_ortho_matrix(near, far, w, h):
        ortho = np.eye(4)
        ortho[0, 0] = 2 / w
        ortho[1, 1] = 2 / h
        ortho[2, 2] = - 2 / (far - near)
        ortho[2, 3] = - (far + near) / (far - near)
        return ortho

    def compute_perspective_matrix(self, fov_h, fov_v, near, far):
        Sh = 1 / np.tan((fov_h/2.0) * (np.pi/180.0))
        Sv = 1 / np.tan((fov_v / 2.0) * (np.pi / 180.0))
        persp = np.eye(4)
        persp[0, 0] = Sh
        persp[1, 1] = Sv

        # Non-linear depth projection
        if self.depth_mode == self.depth_modes_nonlinear:
            persp[2, 2] = -(far + near) / (far - near)
            persp[2, 3] = -(2.0 * far * near) / (far - near)

        # Linear depth projection
        elif self.depth_mode == self.depth_modes_linear:
            persp[2, 2] = -2 / (far - near)        # Linearly scales depth to the [0,2] range
            persp[2, 3] = -near                       # Shifts depth to normalized device coordinates [-1,1]
        else:
            raise ValueError

        persp[3, 3] = 0.0
        persp[3, 2] = -1
        return persp

    def insert_graph(self, nodes):
        assert self.root is not None
        for n in nodes:
            n.id = len(self.nodes)
            self.nodes.append(n)
            if n.parent is None:
                n.set_parent(self.root)

    def clear(self, r=0.0, g=0.2, b=0.2, a=1.0):
        self.ctx.clear(r, g, b, a)

    def delete_graph(self, n):
        if n is not None:
            while len(n.children) > 0:
                self.delete_graph(n.children[0])

            self.delete_node(n)

    def delete_node(self, n):
        if n.parent is not None:
            # Delete the node from its parent list
            for i in range(len(n.parent.children)):
                if n.parent.children[i].id == n.id:
                    del n.parent.children[i]
                    break

            # Re-assign its children to the parent
            for c in n.children:
                c.set_parent(n.parent)

        # Delete from the node list
        for i in range(len(self.nodes)):
            if self.nodes[i].id == n.id:
                del self.nodes[i]
                break

    def draw(self):
        self.ctx.viewport = (0, 0, self.width, self.height)
        # glViewport(0, 0, self.width, self.height)
        aspect = self.width / float(self.height)
        self.perspective = self.compute_perspective_matrix(self.FoV, self.FoV / aspect, self.near, self.far)

        self.ctx.enable(mgl.BLEND)
        self.ctx.enable(mgl.DEPTH_TEST)
        self.root.draw(self.perspective, self.camera.camera_matrix, np.eye(4), self.render_mode)
        # self.swap_buffers()

    # TODO: Enable camera facing text rendering
    # TODO: Enable 2D text scaling
    def draw_text(self, text, pos, color=(1, 1, 1), line_height=20):
        position = list(pos[0:2])
        if not self.glut_init:
            glutInit()
            self.glut_init = True

        glColor3fv(color)
        glWindowPos2dv(position[0:2])
        for ch in text:
            if ch == '\n':
                position[1] = position[1] - line_height
                glWindowPos2dv(position[0:2])
            else:
                glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(ch))

    def draw_line(self, a, b, color, thickness):
        # pa = np.matmul(np.append(a, 1).reshape(1,-1), np.matmul(self.perspective, self.camera.camera_matrix).transpose())
        # pb = np.matmul(np.append(b, 1).reshape(1,-1), np.matmul(self.perspective, self.camera.camera_matrix).transpose())
        # pa = np.matmul(np.append(a, 1), self.perspective)[0:3]
        # pb = np.matmul(np.append(b, 1), self.perspective)[0:3]
        mvp = np.matmul(self.perspective, self.camera.camera_matrix)
        mat = tuple(np.array(mvp, np.float32).reshape(-1, order='F'))
        glUseProgram(0)
        glLoadIdentity()
        glLoadMatrixf(mat)
        glLineWidth(thickness)
        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glBegin(GL_LINES)
        glColor3fv(np.array(color[0:3], np.float32))
        glVertex3fv(a)
        glVertex3fv(b)
        glEnd()
        glEnable(GL_LIGHTING)
        # line = CPointCloud(self.ctx)
        # line.draw_mode = mgl.LINE_STRIP
        # line.set_data(np.concatenate((a, color, b, color)))
        # line.size = thickness
        # line.draw(np.matmul(self.perspective, self.camera.camera_matrix))
        # line.__del__()

    def get_depth_image(self):
        zFar = self.far
        zNear = self.near

        depth_buffer = np.frombuffer(
            self.fbo.read(viewport=self.ctx.viewport, components=1, dtype='f4', attachment=-1),
            dtype=np.dtype('f4')).reshape(self.height, self.width)

        z_ndc = depth_buffer * 2.0 - 1.0  # Convert back to Normalized Device Coordinates [0,1] -> [-1,1]
        if self.depth_mode == self.depth_modes_linear:
            depth_image = z_ndc * (zFar - zNear) + zNear  # Linear inverse depth
        elif self.depth_mode == self.depth_modes_nonlinear:
            depth_image = (2.0 * zNear * zFar) / (zFar + zNear - z_ndc * (zFar-zNear))  # Non-linear inverse depth
        else:
            raise ValueError

        return depth_image

    def process_event(self, event):
        self.camera.process_event(event)

        # Cursor repositioning
        cursor_margin = 10
        if event.type == CEvent.MOUSEMOTION and np.any(event.data[2]):
            if event.data[0][0] > self.width-cursor_margin:
                self.wm.set_mouse_pos(cursor_margin, event.data[0][1])
            if event.data[0][0] < cursor_margin:
                self.wm.set_mouse_pos(self.width - cursor_margin, event.data[0][1])
            if event.data[0][1] > self.height-cursor_margin:
                self.wm.set_mouse_pos(event.data[0][0], cursor_margin)
            if event.data[0][1] < cursor_margin:
                self.wm.set_mouse_pos(event.data[0][0], self.height - cursor_margin)

        if event.type == CEvent.KEYDOWN:
            if event.type == CEvent.QUIT:
                quit()

        if event.type == CEvent.VIDEORESIZE:
            self.wm.set_window_mode((event.data[1], event.data[2]))
            aspect = event.data[1] / float(event.data[2])
            self.perspective = self.compute_perspective_matrix(self.FoV, self.FoV/aspect, self.near, self.far)
            self.width = event.data[1]
            self.height = event.data[2]
            glViewport(0, 0, self.width, self.height)
            print("Window resize (w:%d, h:%d)" % (event.data[1], event.data[2]), event.data[0])

    def __repr__(self):
        return repr(self.root)


class CNode(object):
    def __init__(self, id=0, parent=None, transform=CTransform(), geometry=None, material=None):
        self.id = id
        self.parent = None
        self.children = []
        self.t = transform
        self.geom = geometry
        self.mat = material
        self.pybullet_id = None
        self.pybullet_v_mat = np.eye(4)
        if parent is not None:
            self.set_parent(parent)

    def get_graph_ids(self):
        res = [self.id]
        for c in self.children:
            res.extend(c.get_graph_ids())
        return res

    def set_parent(self, p):
        self.parent = p
        if p is not None:
            p.children.append(self)

    def draw(self, perspective, view, model, mode):
        model = np.matmul(model, self.t.t)
        Mvp = np.matmul(perspective, np.matmul(view, model))
        # Mvp = np.matmul(perspective, view)
        if self.geom is not None:
            self.geom.draw(Mvp, mode)
        for c in self.children:
            c.draw(perspective, view, model, mode)
            # c.draw(perspective, view, np.eye(4), mode)

    def __repr__(self):
        res = ""
        for c in self.children:
            res = res + repr(c)
        if self.parent is not None:
            res = res + "id: %d, p: %d, " % (self.id, self.parent.id) + repr(self.t) + "\n"
        else:
            res = res + "[ROOT] id: %d, " % self.id + repr(self.t) + "\n"
        return res


class CPointCloud(object):
    def __init__(self, ctx, vshader=None, fshader=None):
        self.ctx = ctx
        self.data = []
        self.vbo = None
        self.vao = None
        if vshader is not None:
            self.vertex_shader = open(vshader).read()
        else:
            self.vertex_shader = point_cloud_vertex_shader
        if fshader is not None:
            self.fragment_shader = open(fshader).read()
        else:
            self.fragment_shader = point_cloud_fragment_shader
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)
        self.draw_mode = mgl.POINTS
        self.size = 1

    def set_data(self, data):
        if self.vbo is not None:
            self.vbo.release()
        if self.vao is not None:
            self.vao.release()
        self.data = data
        self.vbo = self.ctx.buffer(data)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 4f', 'in_vert', 'in_color')])

    def draw(self, mvp, mode=mgl.POINTS):
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        self.prog['Mvp'].value = tuple(np.array(mvp, np.float32).reshape(-1, order='F'))
        self.prog['psize'].value = self.size
        if self.draw_mode is not None:
            mode = self.draw_mode
        self.vao.render(mode)
        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE)

    def __del__(self):
        if self.data is not None:
            self.data = None
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.prog is not None:
            self.prog.release()
            self.prog = None


class CGeometry(object):
    def __init__(self, ctx, vshader=None, fshader=None):
        self.ctx = ctx
        self.data = []
        self.vbo = None
        self.vao = None
        self.ibo = None
        if vshader is not None:
            self.vertex_shader = open(vshader).read()
        else:
            self.vertex_shader = geometry_vertex_shader
        if fshader is not None:
            self.fragment_shader = open(fshader).read()
        else:
            self.fragment_shader = geometry_fragment_shader
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)
        self.draw_mode = None
        self.texture = None

    def set_texture(self, path):
        texture_image = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
        texture_image_data = texture_image.tobytes()
        self.texture = self.ctx.texture(size=texture_image.size, components=4, data=texture_image_data)
        self.texture.build_mipmaps()
        self.texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)

    def set_data(self, data, indices=None):
        self.data = data
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self.vbo = self.ctx.buffer(data)
        if self.ibo is not None:
            self.ibo.release()
            self.ibo = None
        if indices is not None:
            self.ibo = self.ctx.buffer(indices)
            self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 3f 3f 4f', 'in_vert', 'in_norm', 'in_text', 'in_color')], index_buffer=self.ibo)
        else:
            self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 3f 3f 4f', 'in_vert', 'in_norm', 'in_text', 'in_color')])

    def draw(self, mvp, mode=mgl.TRIANGLE_STRIP):
        # TODO: Extract lights outside
        self.prog['Light'].value = (1.0, 1.0, 3.0)
        self.prog['Mvp'].value = tuple(np.array(mvp, np.float32).reshape(-1, order='F'))
        tex_id = np.array(0, np.uint16)
        self.prog['Texture'].value = tex_id
        if self.texture is not None:
            self.texture.use(tex_id)

        if self.draw_mode is not None:
            mode = self.draw_mode
        self.vao.render(mode)

    def __del__(self):
        del self.data
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.ibo is not None:
            self.ibo.release()
            self.ibo = None
        if self.prog is not None:
            self.prog.release()
            self.prog = None


class CImage(CGeometry):
    def __init__(self, ctx):
        self.ctx = ctx
        self.data = []
        self.vbo = None
        self.vao = None
        self.ibo = None
        self.vertex_shader = image_vertex_shader
        self.fragment_shader = image_fragment_shader
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)
        self.draw_mode = None
        self.texture = None
        self.offset = (0.0, 0.0)
        self.size = (1.0, 1.0)
        self.color = np.array((0, 0, 0, 1), np.float32)

    def set_position(self, offset, size):
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None

        v0 = np.array((offset[0],           offset[1], 0, 0), np.float32)
        v1 = np.array((offset[0] + size[0], offset[1], 1, 0), np.float32)
        v2 = np.array((offset[0],           offset[1] + size[1], 0, 1), np.float32)
        v3 = np.array((offset[0] + size[0], offset[1] + size[1], 1, 1), np.float32)
        self.data = np.concatenate((v2, self.color,
                                    v1, self.color,
                                    v0, self.color,
                                    v2, self.color,
                                    v3, self.color,
                                    v1, self.color))
        self.vbo = self.ctx.buffer(self.data)
        self.offset = offset
        self.size = size
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f 4f', 'in_vert', 'in_text', 'in_color')])

    def set_texture(self, image):
        if isinstance(image, PIL.Image.Image):
            texture_image = image.convert('RGBA')
            texture_image_data = texture_image.tobytes()

        elif isinstance(image, str):
            texture_image = Image.open(image).transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
            texture_image_data = texture_image.tobytes()
        else:
            raise Exception("Unable to interpret texture type. Required a PIL.Image or a path to an image. Got " + str(type(image)))

        if self.texture is not None:
            self.texture.release()
            self.texture = None

        self.texture = self.ctx.texture(size=texture_image.size, components=4, data=texture_image_data)
        self.texture.build_mipmaps()
        self.texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)

    def draw(self, mvp, mode=mgl.TRIANGLES):
        glDisable(GL_DEPTH_TEST)
        tex_id = np.array(0, np.uint16)
        self.prog['Texture'].value = tex_id
        if self.texture is not None:
            self.texture.use(tex_id)

        if self.draw_mode is not None:
            mode = self.draw_mode
        self.vao.render(mode)
        glEnable(GL_DEPTH_TEST)
