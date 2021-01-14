KBINPUT = False

def kb_capture_thread():
    """Capture when the ENTER key is pressed."""
    global KBINPUT
    input()
    KBINPUT = True