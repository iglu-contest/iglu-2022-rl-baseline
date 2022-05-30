apt-get install x11-utils > /dev/null 2>&1
pip install pyglet > /dev/null 2>&1
apt-get install -y xvfb python-opengl > /dev/null 2>&1
pip install gym pyvirtualdisplay > /dev/null 2>&1
pip install -q lucid>=0.2.3
pip install -q moviepy
pip install pyrender

apt update && apt install xvfb && pip3 install pyvirtualdisplay && pip install pyvirtualdisplay

export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export DISPLAY=:1

gcloud components update -y