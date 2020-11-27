KBINPUT = False

def kb_capture_thread():
    """Capture a keyboard input."""
    global KBINPUT
    input()
    #_ = getch.getch()
    KBINPUT = True