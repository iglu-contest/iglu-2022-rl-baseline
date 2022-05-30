from pyvirtualdisplay import Display
# from selenium import webdriver
# from selenium.webdriver.firefox.options import Options
# Virtual display settings
display = Display (visible = 0, size = (800, 600))
display.start ()

import numpy as np
import json
import moviepy.editor as mvp
from google.colab import files

def init_virtual_display():
    import os
    import pyrender
    if 'PYOPENGL_PLATFORM' not in os.environ:
        from pyrender.platforms.pyglet_platform import PygletPlatform
        _platform = PygletPlatform(500, 600)
    elif os.environ['PYOPENGL_PLATFORM'] == 'egl':
        from pyrender.platforms import egl
        device_id = int(os.environ.get('EGL_DEVICE_ID', '0'))
        egl_device = egl.get_device_by_index(device_id)
        _platform = egl.EGLPlatform(500, 600,
                                    device=egl_device)

    import ctypes
    from ctypes import pointer, util
    import os
    from lucid.misc.gl.glcontext import create_opengl_context

    # Now it's safe to import OpenGL and EGL functions
    import OpenGL.GL as gl

    # create_opengl_context() creates GL context that is attached to an
    # offscreen surface of specified size. Note that rendering to buffers
    # of different size and format is still possible with OpenGL Framebuffers.
    #
    # Users are expected to directly use EGL calls in case more advanced
    # context management is required.
    WIDTH, HEIGHT = 640, 480
    create_opengl_context((WIDTH, HEIGHT))

    # OpenGL context is available here.

    print(gl.glGetString(gl.GL_VERSION))
    print(gl.glGetString(gl.GL_VENDOR))
    # print(gl.glGetString(gl.GL_EXTENSIONS))


init_virtual_display()
init_virtual_display()