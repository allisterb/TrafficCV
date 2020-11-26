import getch
KBINPUT = False

def kb_capture_thread():
    global KBINPUT
    getch.getch()
    KBINPUT = True