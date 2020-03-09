import os
import string
import numpy as np
import transformations as tf
import moderngl as mgl
import PIL
from PIL import Image, ImageFont
from PIL import ImageFilter
import copy

from pyglfw import pyglfw
from pathlib import Path

'''
TODO:
- Implement the rgb+depth image in a single call. Reuse the RGB to add on top helper gizmos and produce the color render
- Shadows
- Add material class
- Fix lighting
- Primitive geometry makers
- Implement transparency handling by ordering the rendered objects by distance to the camera
- Add colormaps

- GUI
    - Screenshot button
    - Focus on cursor point after pressing F        
'''

point_cloud_vertex_shader = ''' 
#version 330 core

uniform mat4 Mvp;

in vec3 in_vert;
in vec4 in_color;

out vec4 v_color;

void main() {
	v_color = in_color;
	gl_Position = Mvp * vec4(in_vert, 1.0);
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

uniform mat4 persp_m;
uniform mat4 view_m;
uniform mat4 model_m;
uniform int normal_colors;

in vec3 in_vert;
in vec3 in_norm;
in vec3 in_text;
in vec4 in_color;

out VS_OUT {
    vec3 v_vert;
    vec3 v_norm;
    vec3 v_text;
    vec4 v_color;
} vs_out;

void main() {
//    mat3 normalMatrix = mat3(transpose(inverse(view_m * model_m)));
    mat3 normalMatrix = mat3(view_m * model_m);
    vs_out.v_vert = in_vert;
    vs_out.v_norm = normalize(vec3(persp_m * vec4(normalMatrix * in_norm, 0.0)));
    vs_out.v_text = in_text;
    if (normal_colors > 0)
    {
        vs_out.v_color = vec4(abs(in_norm), 1);    
    }
    else
    {
        vs_out.v_color = in_color;    
    }
    gl_Position = persp_m * view_m * model_m * vec4(vs_out.v_vert, 1.0);
}
'''

geometry_normals_geometry_shader = '''
#version 330 core
layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

uniform float normal_len;

in VS_OUT {
    vec3 v_vert;
    vec3 v_norm;
    vec3 v_text;
    vec4 v_color;
} gs_in[];

out VS_OUT {
    vec3 v_vert;
    vec3 v_norm;
    vec3 v_text;
    vec4 v_color;
} gs_out;

void GenerateLine(int index)
{
    gs_out.v_vert = gs_in[index].v_vert;
    gl_Position = gl_in[index].gl_Position;
    gs_out.v_norm = gs_in[index].v_norm;
    gs_out.v_text = gs_in[index].v_text;
    gs_out.v_color = gs_in[index].v_color;
    EmitVertex();
    
    float dist = length(gl_Position);
    gs_out.v_vert = gs_in[index].v_vert + gs_in[index].v_norm * normal_len * dist;
    gl_Position = gl_in[index].gl_Position + vec4(gs_in[index].v_norm * normal_len * dist, 0.0);
    gs_out.v_norm = gs_in[index].v_norm;
    gs_out.v_text = gs_in[index].v_text;
    gs_out.v_color = gs_in[index].v_color;
    EmitVertex();
    EndPrimitive();
}

void main()
{
    GenerateLine(0); // first vertex normal
    GenerateLine(1); // second vertex normal
    GenerateLine(2); // third vertex normal
}
'''

geometry_fragment_shader = '''
#version 330 core

uniform int normal_colors;
uniform sampler2D Texture;

in VS_OUT {
    vec3 v_vert;
    vec3 v_norm;
    vec3 v_text;
    vec4 v_color;
} fs_in;

out vec4 f_color;

void main()
{
    vec4 color = texture(Texture, fs_in.v_text.xy);
    if (normal_colors > 0)
    {
        f_color = fs_in.v_color;
    }
    else
    {
        f_color = color + fs_in.v_color;
    }
}
'''

semantic_vertex_shader = '''
#version 330 core

uniform mat4 Mvp;

in vec3 in_vert;
in vec3 in_norm;
in vec3 in_text;
in vec4 in_color;

uniform uint id;
out vec4 v_color;

void main()
{
    // This dummy is to prevent the compiler optimizing out the unused
    // variables, they are not needed but because the data for the meshes
    // is re-used by this shader the data layout has to be the same and
    // include normals, textures and color. 
    vec4 dummy = vec4((in_norm + in_text), 0) + in_color;
     
    v_color = vec4( float(id & 0x000000ffu)/255.0, 
                    float((id & 0x0000ff00u) >> 8)/255.0, 
                    float((id & 0x00ff0000u) >> 16)/255.0, 
                    1.0);
    if ( id < 0u )
    {
        v_color = v_color + dummy * 0.00001;
    }
	gl_Position = Mvp * vec4(in_vert, 1.0);
}
'''

semantic_fragment_shader = '''
#version 330 core

in vec4 v_color;
out vec4 f_color;

void main()
{
    f_color = v_color;
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

# Plot line shaders adapted from
# https://vitaliburkov.wordpress.com/2016/09/17/simple-and-fast-high-quality-antialiased-lines-with-opengl/
plot_vertex_shader = '''
#version 330 core

in vec3 in_vert;
in vec4 in_color;

out vec3 v_vert;
out vec4 v_color;

uniform mat4 Mvp;

void main() {
	v_vert = in_vert;
	v_color = in_color;
    gl_Position = Mvp * vec4(v_vert, 1.0);
}
'''

plot_fragment_shader = '''
#version 330 core
in vec3 v_vert;
in vec4 v_color;
out vec4 f_color;

void main()
{
    f_color = v_color;
}
'''

# default_vertex_shader = semantic_vertex_shader
# default_fragment_shader = semantic_fragment_shader
default_vertex_shader = geometry_vertex_shader
default_fragment_shader = geometry_fragment_shader


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

    def get_position(self):
        return self.t[0:3, 3].reshape(1, 3)

    def set_position(self, trans):
        self.t[0:3, 3] = trans

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
    def __init__(self, alpha=0.7, beta=0.7, distance=2.0, focus=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), width=640, height=480, focal_px=620):
        self.r = distance
        self.alpha = alpha
        self.beta = beta
        self.focus_point = np.array(focus)
        self.up_vector = np.array(up)
        self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
        self.sensitivity = 0.005
        self.focal_px = focal_px
        self.set_intrinsics(width, height, focal_px, focal_px, width / 2, height / 2, 0)

    def __repr__(self):
        res = "Camera"
        res += "\n |- r: %5.3f" % self.r
        res += "\n |- alpha: %5.3f" % self.alpha
        res += "\n |- beta: %5.3f" % self.beta
        res += "\n |- target: [%5.3f, %5.3f, %5.3f]" % (self.focus_point[0], self.focus_point[1], self.focus_point[2])
        res += "\n |- up: [%5.3f, %5.3f, %5.3f]" % (self.up_vector[0], self.up_vector[1], self.up_vector[2])
        res += "\n |- fx, fy, cx, cy: [%5.3f, %5.3f, %5.3f, %5.3f]" % (self.fx, self.fy, self.cx, self.cy)
        res += "\n |- w, h: [%04d, %04d]\n" % (self.width, self.height)
        return res

    def update(self):
        self.camera_matrix = self.look_at(self.focus_point, self.up_vector)

    def set_resolution(self, width, height):
        self.set_intrinsics(width, height, self.focal_px, self.focal_px, width / 2, height / 2, self.s)

    def set_intrinsics(self, width, height, fx, fy, cx, cy, skew):
        self.s = skew
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def process_event(self, event):
        if event.type == CEvent.KEYDOWN:
            if event.data[0] == pyglfw.api.GLFW_KEY_V:
                print("CAMERA PARAMETERS")
                print("Camera: a:", self.alpha, " b:", self.beta, " r:", self.r)
                print("Focus point: ", self.focus_point, " Up vector:", self.up_vector)

            if event.data[0] == pyglfw.api.GLFW_KEY_C:
                self.alpha = 0.0
                self.beta = 0.0
                self.r = 5.0
                self.focus_point = np.array([0.0, 0.0, 0.0])
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
                print("RESET CAMERA")
                print("Camera: a:", self.alpha, " b:", self.beta, " r:", self.r)

        if event.type == CEvent.MOUSEBUTTONDOWN:
            if event.data is None:
                pass
            # Scroll down
            elif event.data[1] == 4:
                self.r = self.r - self.sensitivity*10 if self.r > self.sensitivity*10 else self.sensitivity
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)
            # Scroll up
            elif event.data[1] == 5:
                self.r = self.r + self.sensitivity*10
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)

        if event.type == CEvent.MOUSEMOTION:
            if event.data[2][0] and np.abs(event.data[1][0]) < 50 and np.abs(event.data[1][1]) < 50:
                self.alpha = self.alpha + event.data[1][0] * self.sensitivity
                if self.alpha > 2*np.pi or self.alpha < -2*np.pi:
                    self.alpha = 0

                self.beta = self.beta + event.data[1][1] * self.sensitivity
                if self.beta > np.pi/1.9:
                    self.beta = np.pi/1.9
                elif self.beta < -np.pi/1.9:
                    self.beta = -np.pi/1.9

                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)

            if event.data[2][1] and np.abs(event.data[1][0]) < 50 and np.abs(event.data[1][1]) < 50:
                self.focus_point = self.focus_point - event.data[1][0] * self.camera_matrix[0, 0:3] * self.sensitivity
                self.focus_point = self.focus_point + event.data[1][1] * self.camera_matrix[1, 0:3] * self.sensitivity
                self.camera_matrix = self.look_at(self.focus_point, self.up_vector)

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
        position[0] = self.r * np.cos(alpha) * np.cos(beta) + focus[0]
        position[1] = self.r * np.sin(alpha) * np.cos(beta) + focus[1]
        position[2] = self.r * np.sin(beta) + focus[2]

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
    def init_display(self, fullscreen=False, shared=None):
        raise NotImplementedError()

    def set_window_name(self, name):
        raise NotImplementedError()

    def get_events(self):
        raise NotImplementedError()

    def set_window_mode(self, size, options):
        raise NotImplementedError()

    def set_window_pos(self, pos):
        raise NotImplementedError()

    def set_mouse_pos(self, x, y):
        raise NotImplementedError()

    def get_mouse_pos(self):
        raise NotImplementedError()

    def make_current(self):
        raise NotImplementedError()


class COffscreenWindowManager(CWindowManager):
    def init_display(self, fullscreen=False, shared=None):
        pass

    def set_window_name(self, name):
        pass

    def get_events(self):
        return []

    def set_window_mode(self, size, options):
        pass

    def set_window_pos(self, pos):
        pass

    def set_mouse_pos(self, x, y):
        pass

    def get_mouse_pos(self):
        return [0, 0]

    def draw(self):
        pass

    def close(self):
        pass

    def make_current(self):
        pass


class CGLFWWindowManager(CWindowManager):
    def init_display(self, fullscreen=False, shared=None):

        if not pyglfw.init():
            os.sys.exit(1)

        pyglfw.Window.hint(visible=False)
        pyglfw.Window.hint(samples=4)
        if fullscreen:
            self.window = pyglfw.Window(200, 200, "", pyglfw.get_primary_monitor(), shared=shared)
        else:
            self.window = pyglfw.Window(200, 200, "", None, shared=shared)
        self.window.swap_interval(0)
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
        ev.key = key
        self.event_queue.append(ev)
        # print("keybrd: key=%s scancode=%s action=%s mods=%s" % (key, scancode, action, mods))

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
        self.window.make_current()
        res = copy.deepcopy(self.window.event_queue)
        self.window.event_queue.clear()
        return res

    def make_current(self):
        self.window.make_current()

    def draw(self):
        self.window.make_current()
        self.window.swap_buffers()

    def set_window_mode(self, size, options=None):
        self.window.make_current()
        self.window.size = size

    def set_window_pos(self, pos):
        self.window.make_current()
        self.window.pos = pos

    def set_mouse_pos(self, x, y):
        self.window.cursor_pos = (x, y)

    def get_mouse_pos(self):
        pos = self.window.cursor_pos
        return pos

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None
        # pyglfw.terminate()

############################################################################################################
# END OF WINDOW MANAGER IMPLEMENTATIONS
############################################################################################################


class CScene(object):
    def __init__(self, name="PyViewer", width=800, height=600, location=(0, 0), window_manager=CGLFWWindowManager(), near=0.001, far=100.0, options=None, fullscreen=False, shared=None):

        # The window manager should create the OpenGL context
        self.wm = window_manager
        shared_window = None
        if shared is not None:
            shared_window = shared.wm.window

        self.init_display(name, width, height, location=location, options=options, fullscreen=fullscreen, shared=shared_window)

        print("ModernGL: ", mgl.__version__)
        if isinstance(window_manager, COffscreenWindowManager):
            self.ctx = mgl.create_standalone_context()
        else:
            self.ctx = mgl.create_context()
        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))
        self.active_fbo = "rgb"

        self.width = width
        self.height = height
        self.options = options

        # Create auxiliary frame buffer for semantic segmentation
        self.fbo_seg = self.ctx.simple_framebuffer((self.width, self.height))

        self.ctx.viewport = (0, 0, self.width, self.height)
        self.root = CNode(id=0, parent=None, transform=CTransform(), geometry=None, material=None)
        self.nodes = [self.root]
        self.clear_color = (0.0, 0.2, 0.2, 1.0)

        # TODO: BUG this two lines are a dirty solution of a problem that gets the first inserted node to
        # set the id of the root node. By having a child of the root and building the scene from there it is
        # temporarily fixed
        # dummy_node = CNode(id=None, parent=None, transform=CTransform(), geometry=None, material=None)
        # self.insert_graph([dummy_node])

        self.camera = CCamera(width=width, height=height, focal_px=650)
        self.render_mode = mgl.TRIANGLES

        # aspect = self.width / float(self.height)
        self.near = near
        self.far = far
        self.show_normals = False
        # self.FoV = 60.0
        # self.perspective = self.compute_perspective_matrix(self.FoV, self.FoV / aspect, near, far)

        proj_matrix = self.compute_projection_matrix(self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy,
                                                     self.near, self.far, self.camera.s)

        ndc_matrix = self.compute_ortho_matrix(self.ctx.viewport[0], self.ctx.viewport[2],
                                               self.ctx.viewport[1], self.ctx.viewport[3],
                                               self.near, self.far)

        self.perspective = np.matmul(ndc_matrix, proj_matrix)
        self.perspective = ndc_matrix

        self.set_font(font_path=None, font_size=48)

        self.segment_program = self.ctx.program(vertex_shader=semantic_vertex_shader,
                                                fragment_shader=semantic_fragment_shader)

        self.geometry_program = self.ctx.program(vertex_shader=geometry_vertex_shader,
                                                 fragment_shader=geometry_fragment_shader)

        self.geometry_normals_program = self.ctx.program(vertex_shader=geometry_vertex_shader,
                                                         fragment_shader=geometry_fragment_shader,
                                                         geometry_shader=geometry_normals_geometry_shader)

    def __del__(self):
        # print("CScene::__del__")
        self.delete_graph(self.root)
        self.ctx.finish()
        self.wm.close()
        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
        if self.fbo_seg is not None:
            self.fbo_seg.release()
            self.fbo_seg = None
        # self.ctx.release()

    #############################################################
    # GENERIC INITIALIZATION METHODS. WRAPPER TO HANDLE MULTIPLE WINDOW MANAGERS (pygame, pyglfw, ...)
    # current window manager = glfw
    #############################################################
    def init_display(self, name, width, height, location, options=None, fullscreen=False, shared=None):
        self.wm.init_display(fullscreen=fullscreen, shared=shared)
        self.width = width
        self.height = height
        self.wm.set_window_mode((self.width, self.height), options=options)
        self.wm.set_window_name(name)
        self.wm.set_window_pos(location)

    def get_window_pos(self):
        return self.wm.window.pos

    def set_window_pos(self, pos, monitor=None):
        if monitor is None:
            monitor = pyglfw.get_primary_monitor()
        self.wm.window.pos = np.array(pos) + np.array(monitor.pos)

    def make_current(self):
        self.wm.make_current()
        self.get_active_fbo().use()

    def swap_buffers(self):
        self.make_current()
        if self.ctx.screen is not None:
            self.ctx.copy_framebuffer(self.ctx.screen, self.get_active_fbo())
        self.wm.draw()

    def get_events(self):
        pyglfw.poll_events()
        events = self.wm.get_events()

        return events

    def set_window_name(self, name):
        self.wm.set_window_name(name)

    def set_window_mode(self, size, options):
        self.wm.window.make_current()
        self.ctx.screen.viewport = (0, 0, self.width, self.height)
        self.ctx.screen.scissor = None
        self.fbo_seg = self.ctx.simple_framebuffer(size)
        self.fbo = self.ctx.simple_framebuffer(size)
        return self.wm.set_window_mode(size, options)

    @staticmethod
    def compute_ortho_matrix(left, right, bottom, top, near, far):
        ortho = np.eye(4)
        ortho[0, 0] = 2 / (right-left)
        ortho[1, 1] = 2 / (top-bottom)
        ortho[2, 2] = - 2 / (far - near)

        ortho[0, 3] = - (right + left) / (right - left)
        ortho[1, 3] = - (top + bottom) / (top - bottom)
        ortho[2, 3] = - (far + near) / (far - near)
        return ortho

    @staticmethod
    def compute_projection_matrix(fx, fy, cx, cy, near, far, skew=0):
        proj = np.eye(4)
        proj[0, 0] = fx
        proj[1, 1] = fy
        proj[0, 1] = skew
        proj[0, 2] = -cx
        proj[1, 2] = -cy
        proj[2, 2] = near + far
        proj[2, 3] = near * far

        proj[3, 3] = 0.0
        proj[3, 2] = -1
        return proj

    def insert_graph(self, nodes):
        assert self.root is not None
        for n in nodes:
            if n.id is None:
                n.id = len(self.nodes)
            self.nodes.append(n)
            if n.parent is None:
                CNode.set_parent(node=n, parent=self.root)
            if n.children is not None:
                for c in n.children:
                    self.insert_graph([c])

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

    def get_active_fbo(self):
        if self.active_fbo == "rgb":
            return self.fbo
        elif self.active_fbo == "seg":
            return self.fbo_seg
        else:
            raise ValueError("Unknown active fbo type. Valid values are 'rgb' or 'seg'")

    def set_active_fbo(self, fbo_type="rgb"):
        if fbo_type == "rgb" or fbo_type == "seg":
            self.active_fbo = fbo_type
        else:
            raise ValueError("Unknown fbo type. Valid values are 'rgb' or 'seg'")

    def clear(self, color_rgba=(0.0, 0.2, 0.2, 1.0)):
        self.clear_color = color_rgba
        self.wm.make_current()
        self.ctx.clear(color_rgba[0], color_rgba[1], color_rgba[2], color_rgba[3])

    def draw(self, camera=None, use_ortho=False, fbo=None):
        if fbo is not None:
            self.set_active_fbo(fbo)
        else:
            self.set_active_fbo("rgb")
        self.make_current()

        if camera is None:
            camera = self.camera
        self.ctx.viewport = (0, 0, camera.width, camera.height)

        proj_matrix = self.compute_projection_matrix(camera.fx, camera.fy, camera.cx, camera.cy,
                                                     self.near, self.far, camera.s)

        ndc_matrix = self.compute_ortho_matrix(self.ctx.viewport[0], self.ctx.viewport[2],
                                               self.ctx.viewport[1], self.ctx.viewport[3],
                                               self.near, self.far)

        self.perspective = np.matmul(ndc_matrix, proj_matrix)
        if use_ortho:
            ortho = np.eye(4)
            ortho[0, 0] = (0.001 * camera.height) / camera.r
            ortho[1, 1] = (0.001 * camera.width) / camera.r
            ortho[2, 2] = - 2 / (self.far - self.near)
            ortho[0, 3] = 0
            ortho[1, 3] = 0
            ortho[2, 3] = 0 #- (self.far + self.near) / (self.far - self.near)
            self.root.draw(ortho, camera.camera_matrix, np.eye(4), self.render_mode)
        else:
            self.root.draw(self.perspective, camera.camera_matrix, np.eye(4), self.render_mode)

        # self.ctx.finish()

    def set_font(self, font_path=None, font_size=64, font_color=(255, 255, 255, 255), background_color=(0, 0, 0, 0)):
        self.make_current()
        if font_path is None:
            font_path = str(Path(__file__).resolve().parent) + "/../fonts/FiraCode-Medium.ttf"
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_texture_map, self.font_texture_uv = self.make_font_texture(self.font, font_color, background_color)
        self.char_width, self.line_height = self.font.getsize("A")
        self.text_display = CImage(self)
        self.text_display.draw_always = True
        self.text_display.set_texture(self.font_texture_map.transpose(Image.FLIP_TOP_BOTTOM))

    @staticmethod
    def make_font_texture(font, font_color=(255, 255, 255, 255), background_color=(0, 0, 0, 0)):
        uv_coords = {}
        text = string.printable

        # Generate a texture image with desired background and text colors
        text_width, text_height = font.getsize(text)
        texmap = font.getmask(text, mode='L')
        text_idx = (np.array(texmap, dtype=np.uint8) > 0).reshape((texmap.size[1], texmap.size[0]))
        back_idx = (np.array(texmap, dtype=np.uint8) == 0).reshape((texmap.size[1], texmap.size[0]))
        texarray = np.zeros((texmap.size[1], texmap.size[0], 4), dtype=np.uint8)
        texarray[text_idx,:] = font_color
        texarray[back_idx,:] = background_color

        # Form the image with antialiasing
        teximg = Image.fromarray(np.array(texarray), mode="RGBA").filter(ImageFilter.SMOOTH_MORE)


        # TODO: This assumes fixed width characters, compute per-character width to enable variable width typefaces
        # Compute each character coordinates
        char_width, char_height = font.getsize("A")
        for i, s in enumerate(text):
            v0 = 0
            v1 = 1
            u0 = (char_width * i) / text_width
            u1 = u0 + char_width / text_width
            uv_coords[s] = (u0, v0, u1, v1)

        return teximg, uv_coords

    def draw_text(self, text, pos, scale=1):
        self.set_active_fbo("rgb")
        self.make_current()

        # Get text height and width and transofrm them to NDC
        char_width = self.char_width / self.width
        line_height = self.line_height / self.height

        # Convert the pos in pixels to NDC. This are the positions for the corner vertices
        pos_ndc = (pos[0] / self.width * 2 - 1, pos[1] / self.height * 2 - 1)

        # Compute vertices for the quad (4 vertices for each character)
        line_num = 0
        verts = np.array([]).astype(np.float32)
        ch_pos = 0
        for ch in text:
            if ch == "\n":
                line_num += 1
                ch_pos = 0
                continue
            y0 = pos_ndc[1] - line_num * line_height * scale
            y1 = y0 + line_height * scale
            x0 = ch_pos * char_width * scale + pos_ndc[0]
            x1 = x0 + char_width * scale
            (u0, v0, u1, v1) = self.font_texture_uv[ch]
            p0 = np.array([x0, y0, u0, v0, 0, 0, 0, 0]).astype(np.float32)
            p1 = np.array([x1, y0, u1, v0, 0, 0, 0, 0]).astype(np.float32)
            p2 = np.array([x1, y1, u1, v1, 0, 0, 0, 0]).astype(np.float32)
            p3 = np.array([x0, y1, u0, v1, 0, 0, 0, 0]).astype(np.float32)
            verts = np.concatenate((verts, p0, p1, p2, p2, p3, p0))
            ch_pos += 1

        self.text_display.is_transparent = True
        self.text_display.draw_always = True
        self.text_display.set_data(verts)
        self.text_display.draw(None, None, None)

    def semantic_render(self):
        self.set_active_fbo("seg")
        self.make_current()

        # Set geometry nodes to visible and all Image, Plot, Line and Text nodes not visible and load semantic shaders
        visibility = [False] * len(self.nodes)
        programs = [None] * len(self.nodes)
        for i, n in enumerate(self.nodes):
            if n.geom is not None:
                programs[i] = n.geom.prog
            if n.geom is not None and isinstance(n.geom, CGeometry) and not \
                    isinstance(n.geom, CImage) and not \
                    isinstance(n.geom, CPlot) and not \
                    isinstance(n.geom, CLines) and not \
                    isinstance(n.geom, CFloatingText):
                visibility[i] = n.visible
                n.set_is_visible(True)
                n.geom.prog = self.segment_program
                n.geom.norm_prog = None
                n.geom.update_shader()
            else:
                visibility[i] = n.visible
                n.set_is_visible(False)

        # Render the semantic image
        self.clear((0, 0, 0, 0))  # Clear previous data
        self.draw(fbo="seg")      # Draw semantic image
        # self.ctx.finish()         # Wait untill all draw calls have been executed

        # Restore visibility to the previous state
        self.set_active_fbo("rgb")
        self.make_current()
        for i, n in enumerate(self.nodes):
            n.set_is_visible(visibility[i])
            if n.geom is not None and programs[i] is not None and not isinstance(n.geom, CImage):
                n.geom.prog = programs[i]
                n.geom.norm_prog = self.geometry_normals_program
                n.geom.update_shader()

    def get_depth_image(self):
        self.semantic_render()

        depth_buffer = np.frombuffer(
            self.fbo_seg.read(components=1, dtype='f4', attachment=-1),
            dtype=np.dtype('f4')).reshape(self.fbo_seg.width, self.fbo_seg.height) * 2.0 - 1.0

        # Non-linear inverse depth transformation
        depth_buffer = (2.0 * self.near * self.far) / (self.far + self.near - depth_buffer * (self.far - self.near))
        img = Image.frombuffer('F', (self.fbo.width, self.fbo.height), depth_buffer, 'raw', 'F', 0, -1).transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def get_depth_colormap(self, depth_image, colormap_f):
        image_cm = np.uint8(colormap_f(np.array(depth_image) / self.far) * 255.0)
        texture_image = Image.frombytes("RGBA", depth_image.size, image_cm)
        return texture_image

    def get_render_image(self):
        self.make_current()
        img = Image.frombytes('RGBA', (self.fbo.width, self.fbo.height), self.fbo.read(components=4), 'raw', 'RGBA', 0, -1).transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def get_semantic_image(self):
        self.semantic_render()
        img = Image.frombytes('RGBA', (self.fbo_seg.width, self.fbo_seg.height), self.fbo_seg.read(components=4), 'raw', 'RGBA', 0, -1).transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def get_depth_and_semantic_image(self):
        self.semantic_render()
        img_sem = Image.frombytes('RGBA', (self.fbo_seg.width, self.fbo_seg.height), self.fbo_seg.read(components=4), 'raw', 'RGBA', 0, -1).transpose(Image.FLIP_TOP_BOTTOM)

        depth_buffer = np.frombuffer(
            self.fbo_seg.read(components=1, dtype='f4', attachment=-1),
            dtype=np.dtype('f4')).reshape(self.fbo_seg.width, self.fbo_seg.height) * 2.0 - 1.0

        # Non-linear inverse depth transformation
        depth_buffer = (2.0 * self.near * self.far) / (self.far + self.near - depth_buffer * (self.far - self.near))
        img_depth = Image.frombuffer('F', (self.fbo.width, self.fbo.height), depth_buffer, 'raw', 'F', 0, -1).transpose(Image.FLIP_TOP_BOTTOM)

        return img_depth, img_sem

    def process_event(self, event):
        self.make_current()

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

        # Close program
        if event.type == CEvent.KEYDOWN:
            if event.type == CEvent.QUIT:
                quit()

        # Resize framebuffers
        if event.type == CEvent.VIDEORESIZE:
            self.set_window_mode((event.data[1], event.data[2]), options=self.options)
            self.width = event.data[1]
            self.height = event.data[2]
            self.wm.viewport = (0, 0, self.width, self.height)
            self.camera.set_resolution(self.width, self.height)
            print("Window resize (w:%d, h:%d)" % (event.data[1], event.data[2]), event.data[0])

        if event.type == CEvent.KEYUP and event.data[0] == pyglfw.api.GLFW_KEY_W:
            self.ctx.wireframe = not self.ctx.wireframe

        if event.type == CEvent.KEYUP and event.data[0] == pyglfw.api.GLFW_KEY_N:
            self.show_normals = not self.show_normals

    def __repr__(self):
        res = "=====\n"
        res += "Scene\n"
        res += "=====\n"
        res += repr(self.camera)
        res += "=====\n"
        res += "Graph\n"
        res += "=====\n"
        res += repr(self.root)
        return res


class CNode(object):
    def __init__(self, id=None, parent=None, transform=CTransform(), geometry=None, material=None):
        self.id = id
        self.parent = None
        self.children = []
        self.t = transform
        self.geom = geometry
        self.mat = material
        self.pybullet_id = None
        self.pybullet_v_mat = np.eye(4)
        self.visible = True
        if parent is not None:
            self.set_parent(self, parent)

    def get_graph_ids(self):
        res = [self.id]
        for c in self.children:
            res.extend(c.get_graph_ids())
        return res

    @staticmethod
    def set_parent(node, parent):
        node.parent = parent
        if parent is not None:
            parent.children.append(node)

    def set_is_visible(self, visible):
        self.visible = visible

    def draw(self, perspective, view, model, mode):
        model = np.matmul(model, self.t.t)
        if self.geom is not None and self.visible:
            self.geom.draw(perspective, view, model, mode, id=self.id)
        for c in self.children:
            c.draw(perspective, view, model, mode)

    def __repr__(self):
        res = ""
        for c in self.children:
            res = res + repr(c)
        if self.parent is not None:
            res = res + "id: %03d, p: %03d," % (self.id, self.parent.id) + repr(self.t) + "\n"
        else:
            res = res + "id: %03d, p: N/A," % self.id + repr(self.t) + " [ROOT] \n"
        return res


class CGeometry(object):
    def __init__(self, scene, vshader=None, fshader=None):
        self.scene = scene
        self.scene.make_current()
        self.ctx = scene.ctx
        self.data = []
        self.vbo = None
        self.vao = None
        self.vaon = None
        self.ibo = None
        if vshader is not None:
            self.vertex_shader = open(vshader).read()
        else:
            self.vertex_shader = default_vertex_shader
        if fshader is not None:
            self.fragment_shader = open(fshader).read()
        else:
            self.fragment_shader = default_fragment_shader
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)
        self.norm_prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader, geometry_shader=geometry_normals_geometry_shader)
        self.draw_mode = None
        self.texture = self.ctx.texture(size=(16, 16), components=4, data=np.zeros((16,16,4), dtype=np.uint8).tobytes())
        self.is_transparent = False
        self.draw_always = False

    def set_texture(self, image, build_mipmaps=True):
        self.scene.make_current()
        if isinstance(image, PIL.Image.Image):
            texture_image = image.convert('RGBA')
            texture_image_data = texture_image.tobytes()

        elif isinstance(image, str):
            texture_image = Image.open(image).transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
            texture_image_data = texture_image.tobytes()

        elif isinstance(image, np.ndarray):
            texture_image = PIL.Image.fromarray(image).transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
            texture_image_data = texture_image.tobytes()
        else:
            raise Exception("Unable to interpret texture type. Required a PIL.Image or a path to an image. Got " + str(type(image)))

        if self.texture is not None:
            self.texture.release()
            self.texture = None

        self.texture = self.ctx.texture(size=texture_image.size, components=4, data=texture_image_data)
        if build_mipmaps:
            self.texture.build_mipmaps()
            self.texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        else:
            self.texture.filter = (mgl.LINEAR, mgl.LINEAR)

    def update_shader(self):
        self.scene.make_current()
        if self.vbo is None:
            return
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.vaon is not None:
            self.vaon.release()
            self.vaon = None
        if self.ibo is not None:
            self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 3f 3f 4f', 'in_vert', 'in_norm', 'in_text', 'in_color')], index_buffer=self.ibo)
            if self.norm_prog is not None:
                self.vaon = self.ctx.vertex_array(self.norm_prog, [(self.vbo, '3f 3f 3f 4f', 'in_vert', 'in_norm', 'in_text', 'in_color')], index_buffer=self.ibo)
        else:
            self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 3f 3f 4f', 'in_vert', 'in_norm', 'in_text', 'in_color')])
            if self.norm_prog is not None:
                self.vaon = self.ctx.vertex_array(self.norm_prog, [(self.vbo, '3f 3f 3f 4f', 'in_vert', 'in_norm', 'in_text', 'in_color')])

    def set_data(self, data, indices=None):
        self.scene.make_current()
        if data is None or len(data) == 0:
            print("Warning :: " + str(__class__) + ". Setting empty data.")
            return

        self.data = data
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        self.vbo = self.ctx.buffer(data)
        if self.ibo is not None:
            self.ibo.release()
            self.ibo = None
        if indices is not None:
            self.ibo = self.ctx.buffer(indices)
        self.update_shader()

    def set_uniforms(self, prog, perspective, view, model, id, tex_id):
        if 'Light' in prog:
            prog['Light'].value = self.scene.light

        if 'Mvp' in prog:
            mvp = np.matmul(perspective, np.matmul(view, model))
            prog['Mvp'].value = tuple(np.array(mvp, np.float32).reshape(-1, order='F'))

        if 'persp_m' in prog:
            prog['persp_m'].value = tuple(np.array(perspective, np.float32).reshape(-1, order='F'))

        if 'view_m' in prog:
            prog['view_m'].value = tuple(np.array(view, np.float32).reshape(-1, order='F'))

        if 'model_m' in prog:
            prog['model_m'].value = tuple(np.array(model, np.float32).reshape(-1, order='F'))

        if 'Texture' in prog:
            prog['Texture'].value = tex_id

        if 'id' in prog:
            prog['id'].value = id & 0xffffffff

    def draw(self, perspective, view, model, mode=mgl.TRIANGLE_STRIP, id=0):
        self.scene.make_current()
        if self.data is None or len(self.data) == 0:
            return

        tex_id = np.array(0, np.uint16)
        self.set_uniforms(self.prog, perspective, view, model, id, tex_id)
        if self.norm_prog is not None:
            self.set_uniforms(self.norm_prog, perspective, view, model, id, tex_id)
            if 'normal_len' in self.norm_prog:
                self.norm_prog['normal_len'].value = 0.03
            if 'normal_colors' in self.norm_prog:
                self.norm_prog['normal_colors'].value = int(self.scene.show_normals)

        if self.draw_mode is not None:
            mode = self.draw_mode

        if self.texture is not None:
            self.texture.use(tex_id)

        if self.draw_always:
            self.ctx.disable(mgl.DEPTH_TEST)
        else:
            self.ctx.enable(mgl.DEPTH_TEST)
        if self.is_transparent:
            self.ctx.enable(mgl.BLEND)
            self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
            self.vao.render(mode)
            if self.scene.show_normals and self.vaon is not None:
                self.ctx.enable(mgl.DEPTH_TEST)
                self.vaon.render(mode)
        else:
            self.ctx.disable(mgl.BLEND)
            self.vao.render(mode)
            if self.scene.show_normals and self.vaon is not None:
                self.ctx.enable(mgl.DEPTH_TEST)
                self.vaon.render(mode)

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
        if self.texture is not None:
            self.texture.release()
            self.texture = None

        # TODO: This needs fixing to release the shader if it is a local shader
        # if self.prog is not None:
        #     self.prog.release()
        #     self.prog = None


class CPointCloud(object):
    def __init__(self, scene, vshader=None, fshader=None):
        self.scene = scene
        self.scene.make_current()
        self.ctx = scene.ctx
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

    def update_shader(self):
        self.scene.make_current()
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 4f', 'in_vert', 'in_color')])

    def set_data(self, data):
        self.scene.make_current()
        if data is None or len(data) == 0:
            print("Warning :: " + str(__class__) + ". Setting empty data.")
            return

        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        self.data = data
        if data is not None:
            self.vbo = self.ctx.buffer(data)
            self.update_shader()

    def draw(self, perspective, view, model, mode=mgl.POINTS, id=0):
        self.scene.make_current()
        if self.data is None or len(self.data) == 0:
            return

        self.ctx.disable(mgl.BLEND)
        self.ctx.enable(mgl.DEPTH_TEST)
        mvp = np.matmul(perspective, np.matmul(view, model))
        self.prog['Mvp'].value = tuple(np.array(mvp, np.float32).reshape(-1, order='F'))
        if self.draw_mode is not None:
            mode = self.draw_mode
        self.ctx.point_size = self.size
        self.ctx.line_width = self.size
        self.vao.render(mode)

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


class CFloatingText(CGeometry):
    def __init__(self, scene, text="text", position=(0, 0, 0), height=0.1):
        super().__init__(scene)
        self.font = None
        self.font_texture_map = None
        self.font_texture_uv = None
        self.char_width = None
        self.line_height = None
        self.draw_mode = mgl.TRIANGLES
        self.position = position
        self.text = text
        self.height = height
        self.aspect_ratio = 1
        self.width = self.aspect_ratio * self.height
        self.set_font()
        self.camera_facing = False
        self.is_transparent = True

    def set_font(self, font_path=None, font_size=64, font_color=(255, 255, 255, 255), background_color=(0, 0, 0, 0)):
        self.scene.make_current()
        if font_path is None:
            font_path = str(Path(__file__).resolve().parent) + "/../fonts/FiraCode-Medium.ttf"
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_texture_map, self.font_texture_uv = CScene.make_font_texture(self.font, font_color, background_color)
        self.char_width, self.line_height = self.font.getsize("A")
        self.set_texture(self.font_texture_map.transpose(Image.FLIP_TOP_BOTTOM), build_mipmaps=False)
        self.aspect_ratio = self.line_height / self.char_width
        self.width = self.height / self.aspect_ratio
        self.update_vertices()
        # self.font_texture_map.show()

    def set_text(self, text):
        self.text = text
        self.update_vertices()

    def set_position(self, pos):
        self.position = pos
        self.update_vertices()

    def set_height(self, height):
        self.height = height
        self.width = self.height / self.aspect_ratio
        self.update_vertices()

    def update_vertices(self):
        verts = np.array([]).astype(np.float32)
        line_num = 0
        ch_pos = 0
        for ch in self.text:
            if ch == "\n":
                line_num += 1
                ch_pos = 0
                continue

            y0 = line_num * self.height
            y1 = y0 + self.height
            x0 = ch_pos * self.width
            x1 = x0 + self.width
            (u0, v0, u1, v1) = self.font_texture_uv[ch]
            # (u0, v0, u1, v1) = (0, 0, 1, 1)
            # Vertex format is 3f (pos) 3f (normal) 3f (texture) 4f (color)
            p0 = np.array([x0, y0, 0, 0, 0, 1, u0, v0, 0, 0, 0, 0, 0]).astype(np.float32)
            p1 = np.array([x1, y0, 0, 0, 0, 1, u1, v0, 0, 0, 0, 0, 0]).astype(np.float32)
            p2 = np.array([x1, y1, 0, 0, 0, 1, u1, v1, 0, 0, 0, 0, 0]).astype(np.float32)
            p3 = np.array([x0, y1, 0, 0, 0, 1, u0, v1, 0, 0, 0, 0, 0]).astype(np.float32)
            verts = np.concatenate((verts, p0, p1, p2, p2, p3, p0))
            ch_pos += 1
        self.set_data(verts)

    def draw(self, perspective, view, model, mode=mgl.TRIANGLES, id=0):
        if self.camera_facing:
            # Compute camera facing rotation matrix
            pass

        super().draw(perspective, view, model, mode, id)


class CImage(CGeometry):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.vertex_shader = image_vertex_shader
        self.fragment_shader = image_fragment_shader
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)
        self.norm_prog = None
        self.draw_mode = mgl.TRIANGLES
        self.texture = None
        self.offset = (0.0, 0.0)
        self.size = (1.0, 1.0)
        self.color = np.array((0, 0, 0, 0), np.float32)
        self.is_transparent = True
        self.draw_always = True

    def set_position(self, offset, size):
        self.scene.make_current()
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
        self.update_shader()

    def set_texture(self, image, build_mipmaps=False):
        super().set_texture(image, build_mipmaps)

    def draw(self, perspective, view, model, mode=mgl.TRIANGLES, id=0):
        super().draw(perspective, view, model, mode, id)

    def update_shader(self):
        self.scene.make_current()
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f 4f', 'in_vert', 'in_text', 'in_color')])

    def set_data(self, data, indices=None):
        self.scene.make_current()
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self.data = data
        self.vbo = self.ctx.buffer(data)
        self.update_shader()


class CLines(CGeometry):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.draw_mode = mgl.LINES
        self.vertex_shader = plot_vertex_shader
        self.fragment_shader = plot_fragment_shader
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)
        self.line_color = np.array([0,0,0,1])
        self.line_width = 1
        self.data = np.array([], dtype=np.float32)

    def update_shader(self):
        self.scene.make_current()
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 4f', 'in_vert', 'in_color')])

    def set_data(self, data, indices=None):
        self.scene.make_current()
        if data is None or len(data) == 0:
            print("Warning :: " + str(__class__) + ". Setting empty data.")
            return

        self.data = data
        self.vbo = self.ctx.buffer(self.data)
        self.update_shader()

    def draw(self, perspective, view, model, mode=mgl.LINES, id=0):
        self.scene.make_current()
        self.ctx.line_width = self.line_width
        super().draw(perspective, view, model, mode, id)


class CPlot(CGeometry):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.draw_mode = mgl.LINES
        self.vertex_shader = plot_vertex_shader
        self.fragment_shader = plot_fragment_shader
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)
        self.line_color = np.array([0,0,0,1])
        self.xlim = [None, None]
        self.ylim = [None, None]
        self.xticks = 10
        self.yticks = 10
        self.xlines = 0
        self.ylines = 0
        self.verts_frame = np.array([], dtype=np.float32)
        self.verts_ticks = np.array([], dtype=np.float32)
        self.verts_label = np.array([], dtype=np.float32)
        self.verts_data  = np.array([], dtype=np.float32)
        self.verts_tick_lines = np.array([], dtype=np.float32)
        self.verts_markers = np.array([], dtype=np.float32)
        self.verts_labels = np.array([], dtype=np.float32)
        self.is_transparent = True
        self.data = np.array([], dtype=np.float32)
        self.data_x = np.array([], dtype=np.float32)
        self.data_y = np.array([], dtype=np.float32)
        self.xlabel = ""
        self.ylabel = ""
        self.title = ""
        self.markers = ""
        self.line_width = 1
        self.blend_factor = 1.5
        self.make_frame()
        self.make_ticks()
        self.make_tick_lines()

    def set_x_label(self, label, c=(0, 0, 0, 1)):
        self.xlabel = label
        self.make_labels(c=c)

    def set_y_label(self, label, c=(0, 0, 0, 1)):
        self.xlabel = label
        self.make_labels(c=c)

    def set_title(self, label, c=(0, 0, 0, 1)):
        self.xlabel = label
        self.make_labels(c=c)

    def set_x_ticks(self, num, c=(0, 0, 0, 0.2)):
        self.xticks = num
        self.make_ticks(c=c)

    def set_y_ticks(self, num, c=(0, 0, 0, 0.2)):
        self.yticks = num
        self.make_ticks(c=c)

    def set_x_lines(self, num, c=(0, 0, 0, 0.2)):
        self.xlines = num
        self.make_tick_lines(c=c)

    def set_y_lines(self, num, c=(0, 0, 0, 0.2)):
        self.ylines = num
        self.make_tick_lines(c=c)

    def set_x_lim(self, min, max):
        self.xlim = (min, max)
        if len(self.data_x) > 0:
            self.make_data()

    def set_y_lim(self, min, max):
        self.ylim = (min, max)
        if len(self.data_x) > 0:
            self.make_data()

    def set_markers(self, markers="", c=(0, 0, 0, 0.2)):
        self.markers = markers
        self.make_markers(c=c)

    def make_labels(self, c):
        self.xlabel
        self.ylabel
        self.title

    # TODO: Integrate this when the axis rescale and in the initialization of the plot
    def make_tick_labels(self, c=(0, 0, 0, 1), format="%5.3f"):
        # Get plotted data limits
        min_x = self.xlim[0] if self.xlim[0] is not None else np.min(self.data_x)
        max_x = self.xlim[1] if self.xlim[1] is not None else np.max(self.data_x)
        min_y = self.ylim[0] if self.ylim[0] is not None else np.min(self.data_y)
        max_y = self.ylim[1] if self.ylim[1] is not None else np.max(self.data_y)

        # Compute the normalized positions for each label
        x_tick_pos = [(max_x-min_x)/i + min_x for i in range(self.xticks)]
        y_tick_pos = [(max_y-min_y)/i + min_y for i in range(self.yticks)]

        x_tick_normpos = [(tick-min_x)/(max_x-min_x) for tick in x_tick_pos]
        y_tick_normpos = [(tick-min_y)/(max_y-min_y) for tick in y_tick_pos]

        # Generate strings for each label and their positions
        x_tick_labels = [format % tick for tick in x_tick_pos]
        y_tick_labels = [format % tick for tick in y_tick_pos]

        # Generate labels at the desired location
        x_text_labels = [self.generate_label(label, [pos, 0], c) for label, pos in zip(x_tick_labels, x_tick_normpos)]
        y_text_labels = [self.generate_label(label, [0, pos], c) for label, pos in zip(y_tick_labels, y_tick_normpos)]

        self.verts_labels = np.concatenate(x_text_labels, y_text_labels)

    # TODO: Implement label generation in normalized coordinates
    # TODO: Implement line breaks
    @staticmethod
    def generate_label(label, size, color):
        vertices = np.array([], dtype=np.float32)
        c_pos = 0.0
        for char in label:
            c_points = CPlot.get_traced_char(char) * size   # Get a charcter in normalized coordinates and scale it
            c_points[:, 0] += c_pos                         # Advance the character in the X direction to the current position
            c_pos = np.max(c_points[:, 0])                  # Increase the current cursor position

            # Add color data to each vertex of the current char being added
            for i in range(0, len(c_points), 3):
                vertices = np.concatenate((vertices, c_points[i:i+3], color))

        return vertices

    # TODO: Implement traced characters lookup table. This must returns 3D vertices of the desired char
    @staticmethod
    def get_traced_char(char):
        vertices = np.array([], dtype=np.float32)
        return vertices

    def make_tick_lines(self, c=None):
        if c is None:
            c = self.line_color

        self.verts_tick_lines = np.array([], dtype=np.float32)

        # X lines
        if self.xlines > 0:
            tick_inc = 1.0 / self.xlines
            for i in range(self.xlines):
                p0 = np.array([tick_inc * i + tick_inc * 0.5, 0], dtype=np.float32)
                p1 = np.array([tick_inc * i + tick_inc * 0.5, 1], dtype=np.float32)
                line = self.make_line(p0, p1, np.array([0, c[0], c[1], c[2], c[3]], dtype=np.float32))
                self.verts_ticks = np.concatenate((self.verts_ticks, line))

        # Y lines
        if self.ylines > 0:
            tick_inc = 1.0 / self.ylines
            for i in range(self.ylines):
                p0 = np.array([0, tick_inc * i + tick_inc*0.5], dtype=np.float32)
                p1 = np.array([1, tick_inc * i + tick_inc*0.5], dtype=np.float32)
                line = self.make_line(p0, p1, np.array([0, c[0], c[1], c[2], c[3]], dtype=np.float32))
                self.verts_ticks = np.concatenate((self.verts_ticks, line))

    def make_ticks(self, c=None, tick_len=0.05, symmetric=False):
        if c is None:
            c = self.line_color

        self.verts_ticks = np.array([], dtype=np.float32)
        # X ticks
        if self.xticks > 0:
            tick_inc = 1.0 / self.xticks
            for i in range(self.xticks):
                p0 = np.array([tick_inc * i + tick_inc * 0.5, -tick_len / 2 * int(symmetric)], dtype=np.float32)
                p1 = np.array([tick_inc * i + tick_inc * 0.5,  tick_len / 2], dtype=np.float32)
                line = self.make_line(p0, p1, np.array([0, c[0], c[1], c[2], c[3]], dtype=np.float32))
                self.verts_ticks = np.concatenate((self.verts_ticks, line))

        # Y ticks
        if self.yticks > 0:
            tick_inc = 1.0 / self.yticks
            for i in range(self.yticks):
                p0 = np.array([-tick_len / 2 * int(symmetric), tick_inc * i + tick_inc * 0.5], dtype=np.float32)
                p1 = np.array([tick_len / 2, tick_inc * i + tick_inc * 0.5], dtype=np.float32)
                line = self.make_line(p0, p1, np.array([0, c[0], c[1], c[2], c[3]], dtype=np.float32))
                self.verts_ticks = np.concatenate((self.verts_ticks, line))

    def make_frame(self, c=None):
        if c is None:
            c = self.line_color

        # Vertex format is 3f (pos) 4f (color)
        color_data = np.array([0, c[0], c[1], c[2], c[3]], dtype=np.float32)
        l1 = self.make_line(np.array([0, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32), color_data)
        l2 = self.make_line(np.array([0, 1], dtype=np.float32), np.array([1, 1], dtype=np.float32), color_data)
        l3 = self.make_line(np.array([1, 1], dtype=np.float32), np.array([1, 0], dtype=np.float32), color_data)
        l4 = self.make_line(np.array([1, 0], dtype=np.float32), np.array([0, 0], dtype=np.float32), color_data)
        self.verts_frame = np.concatenate((l1, l2, l3, l4))

    def make_data(self, c=(0, 0, 0, 1)):
        raise NotImplementedError

    def set_vline(self, x, c=(1, 0, 0, 1)):
        min_x = self.xlim[0] if self.xlim[0] is not None else np.min(self.data_x)
        max_x = self.xlim[1] if self.xlim[1] is not None else np.max(self.data_x)
        x_norm = (x-min_x) / (max_x-min_x)
        color_data = np.array([0, c[0], c[1], c[2], c[3]], dtype=np.float32)
        l1 = self.make_line(np.array([x_norm, 0], dtype=np.float32), np.array([x_norm, 1], dtype=np.float32), color_data)
        self.verts_markers = l1

    def update_shader(self):
        self.scene.make_current()
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 4f', 'in_vert', 'in_color')])

    def draw(self, perspective, view, model, mode=mgl.LINES, id=0):
        self.scene.make_current()
        self.ctx.line_width = self.line_width

        self.data = np.concatenate((self.verts_markers, self.verts_data, self.verts_frame, self.verts_ticks, self.verts_tick_lines, self.verts_label))
        if len(self.data) > 0:
            self.vbo = self.ctx.buffer(self.data)
            self.update_shader()
            super().draw(perspective, view, model, mode, id)

    # This version is just with points (for GL_LINES)
    @staticmethod
    def make_line(p1, p2, vert_attr=np.array([], dtype=np.float32)):
        return np.concatenate((p1, vert_attr, p2, vert_attr))

    # # This version is with triangles (for GL_TRIANGLES)
    # @staticmethod
    # def triangulate_line(p1, p2, thickness=0.01, vert_attr=np.array([], dtype=np.float32)):
    #     ndir = (p1 - p2) / np.linalg.norm(p1 - p2)
    #     ndir = np.array([-ndir[1], ndir[0]], dtype=np.float32)
    #     seg_p1 = p1 + ndir * thickness
    #     seg_p2 = p1 - ndir * thickness
    #     seg_p3 = p2 + ndir * thickness
    #     seg_p4 = p2 - ndir * thickness
    #     return np.concatenate((seg_p1, vert_attr, seg_p2, vert_attr, seg_p3, vert_attr, seg_p2, vert_attr, seg_p3, vert_attr, seg_p4, vert_attr))


class CLinePlot(CPlot):
    def __init__(self, ctx):
        super().__init__(ctx)

    def plot(self, x, y, c=(0, 0, 0, 1)):
        if len(x) <= 1:
            print("ERROR: CLinePlot plot needs more than 1 element.")
            return

        self.data_x = np.array(x)
        self.data_y = np.array(y)

        min_x = self.xlim[0] if self.xlim[0] is not None else np.min(self.data_x)
        max_x = self.xlim[1] if self.xlim[1] is not None else np.max(self.data_x)
        min_y = self.ylim[0] if self.ylim[0] is not None else np.min(self.data_y)
        max_y = self.ylim[1] if self.ylim[1] is not None else np.max(self.data_y)

        # Copy only data that is in range
        filter_idx_x = np.logical_and(self.data_x > min_x, self.data_x < max_x)
        filter_idx_y = np.logical_and(self.data_y > min_y, self.data_y < max_y)
        filter = np.logical_and(filter_idx_x, filter_idx_y)

        self.data_x = self.data_x[filter]
        self.data_y = self.data_y[filter]
        self.make_data(c=c)

    def plot_append(self, x, y, c=(0, 0, 0, 1)):
        data_x = np.array([x])
        data_y = np.array([y])

        # Filter the data point in the plot range
        if len(self.data_x) > 0 and len(self.data_y) > 0:
            min_x = self.xlim[0] if self.xlim[0] is not None else np.min(np.concatenate((self.data_x, data_x)))
            max_x = self.xlim[1] if self.xlim[1] is not None else np.max(np.concatenate((self.data_x, data_x)))
            min_y = self.ylim[0] if self.ylim[0] is not None else np.min(np.concatenate((self.data_y, data_y)))
            max_y = self.ylim[1] if self.ylim[1] is not None else np.max(np.concatenate((self.data_y, data_y)))
        else:
            min_x = self.xlim[0] if self.xlim[0] is not None else np.min(data_x)
            max_x = self.xlim[1] if self.xlim[1] is not None else np.max(data_x)
            min_y = self.ylim[0] if self.ylim[0] is not None else np.min(data_y)
            max_y = self.ylim[1] if self.ylim[1] is not None else np.max(data_y)

        filter_idx_x = np.logical_and(data_x >= min_x, data_x <= max_x)
        filter_idx_y = np.logical_and(data_y >= min_y, data_y <= max_y)
        filter = np.logical_and(filter_idx_x, filter_idx_y)

        data_x = data_x[filter]
        data_y = data_y[filter]

        # Add in range data points to the plot vertex data
        if len(data_x) == 1:
            point = np.array([(data_x[0]-min_x) / (max_x-min_x), (data_y[0]-min_y) / (max_y-min_y), 0.001, c[0], c[1], c[2], c[3]], dtype=np.float32)
            if len(self.verts_data) > 0:
                self.verts_data = self.add_point_to_line(self.verts_data, point)
            else:
                self.verts_data = np.concatenate((point, point))
        elif len(data_x) > 1:
            new_verts_data = self.make_vertex_data(data_x, data_y, c)
            self.verts_data = self.join_lines(self.verts_data, new_verts_data)
        else:
            return

        # Add in range data points to the plot data
        self.data_x = np.concatenate((self.data_x, data_x))
        self.data_y = np.concatenate((self.data_y, data_y))

    @staticmethod
    def join_lines(l1, l2):
        return np.append((l1, l1[-7:], l2[0:8], l2))

    @staticmethod
    def add_point_to_line(line, point):
        return np.concatenate((line, line[-7:], point))

    def make_data(self, c=(0, 0, 0, 1)):
        if len(self.data_x) > 1:
            self.verts_data = self.make_vertex_data(self.data_x, self.data_y, c)

    def make_vertex_data(self, x=np.array([], dtype=np.float32), y=np.array([], dtype=np.float32), c=(0, 0, 0, 1)):
        data = np.array([], np.float32)

        if len(x) <= 0 and len(y) <= 0:
            return data

        if len(self.data_x) > 0 and len(self.data_y) > 0:
            min_x = self.xlim[0] if self.xlim[0] is not None else np.min(self.data_x)
            max_x = self.xlim[1] if self.xlim[1] is not None else np.max(self.data_x)
            min_y = self.ylim[0] if self.ylim[0] is not None else np.min(self.data_y)
            max_y = self.ylim[1] if self.ylim[1] is not None else np.max(self.data_y)
        else:
            min_x = self.xlim[0] if self.xlim[0] is not None else np.min(x)
            max_x = self.xlim[1] if self.xlim[1] is not None else np.max(x)
            min_y = self.ylim[0] if self.ylim[0] is not None else np.min(y)
            max_y = self.ylim[1] if self.ylim[1] is not None else np.max(y)

        range_x = (max_x - min_x)
        range_y = (max_y - min_y)

        if range_x > 0 and range_y > 0:
            # data = np.array([], dtype=np.float32)
            # for i in range(len(x)-1):
            #     # Get the segment coordinates
            #     p1 = np.array([(x[i] - min_x) / (max_x - min_x), (y[i] - min_y) / (max_y - min_y)], dtype=np.float32)
            #     p2 = np.array([(x[i+1] - min_x) / (max_x - min_x), (y[i+1] - min_y) / (max_y - min_y)], dtype=np.float32)
            #
            #     # Get the segment normal and create the 4 line vertices
            #     triangles = self.triangulate_line(p1, p2, self.line_width, color_data)
            #
            #     # Add segment triangles
            #     data = np.concatenate((data, triangles))

            if len(x) == 1 and len(self.data_x) > 0 and len(self.data_y) > 0:
                prev_point = np.array([(self.data_x[-1] - min_x) / (max_x - min_x), (self.data_y[-1] - min_y) / (max_y - min_y), 0.001,c[0], c[1], c[2]], dtype=np.float32)
                data = np.array([(self.data_x[-1]-min_x) / (max_x-min_x), (self.data_y[-1]-min_y) / (max_y-min_y), 0.001, c[0], c[1], c[2], c[3], (x[0]-min_x) / (max_x-min_x), (y[0]-min_y) / (max_y-min_y), 0.001, c[0], c[1], c[2], c[3]], dtype=np.float32)
                data = np.concatenate((prev_point, data[0], data)).flatten()
            else:
                data = np.array([[(x[i]-min_x) / (max_x-min_x), (y[i]-min_y) / (max_y-min_y), 0.001, c[0], c[1], c[2], c[3], (x[i+1]-min_x) / (max_x-min_x), (y[i+1]-min_y) / (max_y-min_y), 0.001, c[0], c[1], c[2], c[3]] for i in range(len(x)-1)], dtype=np.float32)
            return data.flatten()
        else:
            print("ERROR: CLinePlot range of values is 0 for X or Y axis.")
            return data


class CBarPlot(CPlot):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.draw_mode = mgl.TRIANGLES
        self.data_std = np.array([], dtype=np.float32)

    # This version is with triangles (for GL_TRIANGLES)
    @staticmethod
    def triangulate_line(p1, p2, thickness=0.01, vert_attr=np.array([], dtype=np.float32)):
        ndir = (p1 - p2) / np.linalg.norm(p1 - p2)
        ndir = np.array([-ndir[1], ndir[0]], dtype=np.float32)
        seg_p1 = p1 + ndir * thickness * 0.5
        seg_p2 = p1 - ndir * thickness * 0.5
        seg_p3 = p2 + ndir * thickness * 0.5
        seg_p4 = p2 - ndir * thickness * 0.5
        return np.concatenate((seg_p1, vert_attr, seg_p2, vert_attr, seg_p3, vert_attr, seg_p2, vert_attr, seg_p3, vert_attr, seg_p4, vert_attr))

    def plot(self, y, stdev, c=(0, 0, 0, 1)):
        if len(y) <= 0 or len(stdev) <= 0:
            print("ERROR: CLinePlot plot needs more than 0 elements.")
            return

        self.data_y = np.array(y, dtype=np.float32)
        self.data_std = np.array(stdev, dtype=np.float32)

        min_y = self.ylim[0] if self.ylim[0] is not None else np.min(self.data_y - stdev)
        max_y = self.ylim[1] if self.ylim[1] is not None else np.max(self.data_y + stdev)

        # Copy only data that is in range
        filter_idx_y = np.logical_and(self.data_y > min_y, self.data_y < max_y)

        self.data_y = self.data_y[filter_idx_y]
        self.data_std = self.data_std[filter_idx_y]

        # One horizontal tick per bar
        self.set_x_ticks(len(self.data_y))

        self.make_data(c=c)

    def make_data(self, c=(0, 0, 0, 1)):
        if len(self.data_y) > 0:
            self.verts_data = self.make_vertex_data(self.data_y, self.data_std, c)

    def make_vertex_data(self, y, std, c=(0, 0, 0, 1)):
        data = np.array([], np.float32)

        if len(y) <= 0 or len(std) <= 0:
            return data

        self.data_y = y
        self.data_std = std

        min_y = self.ylim[0] if self.ylim[0] is not None else np.min(self.data_y - self.data_std)
        max_y = self.ylim[1] if self.ylim[1] is not None else np.max(self.data_y + self.data_std)

        range_y = (max_y - min_y)

        color_data = np.array([0, c[0], c[1], c[2], c[3]], dtype=np.float32)

        if range_y > 0:
            data = np.array([], dtype=np.float32)
            width = 1/len(self.data_y)
            for i in range(len(self.data_y)):
                # Get the segment coordinates
                x = i / len(self.data_y)
                p1 = np.array([x + width * 0.5, 0], dtype=np.float32)
                p2 = np.array([x + width * 0.5, (y[i] - min_y) / (max_y - min_y)], dtype=np.float32)

                # Get the segment normal and create the 4 line vertices
                triangles = self.triangulate_line(p1, p2, width - 0.01, color_data)

                # Add segment triangles
                data = np.concatenate((data, triangles))
            return data.flatten()
        else:
            print("ERROR: CLinePlot range of values is 0 for X or Y axis.")
            return data

    def draw(self, perspective, view, model, mode=mgl.LINES, id=0):
        self.scene.make_current()
        # Draw first lines
        self.draw_mode = mgl.LINES
        self.ctx.line_width = self.line_width
        self.data = np.concatenate((self.verts_markers, self.verts_frame, self.verts_ticks, self.verts_tick_lines, self.verts_label))
        if len(self.data) > 0:
            self.vbo = self.ctx.buffer(self.data)
            self.update_shader()
            CGeometry.draw(self, perspective, view, model, mode, id)

        # Draw second triangles
        self.draw_mode = mgl.TRIANGLES
        if len(self.verts_data) > 0:
            self.vbo = self.ctx.buffer(self.verts_data)
            self.update_shader()
            CGeometry.draw(self, perspective, view, model, mode, id)
