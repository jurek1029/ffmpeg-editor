#error,warning,info
DebugLeves = {0:"Error",
1:"Warning",
2:"Info"}
DEBUG_LEVEL = 2

def setLevel(lvl):
    global DEBUG_LEVEL
    DEBUG_LEVEL = lvl

def printe(s):
    if DEBUG_LEVEL >= 0: 
        print(f'\33[7m\33[49m\33[91m[{DebugLeves[2]}] {s}\33[0m\33[49m\33[39m')

def printw(s):
    if DEBUG_LEVEL >= 1: 
        print(f'\33[7m\33[49m\33[93m[{DebugLeves[1]}] {s}\33[0m\33[49m\33[39m')
def printi(s):
    if DEBUG_LEVEL >= 2: 
        print(f'\33[4m\33[49m\33[92m[{DebugLeves[2]}] {s}\33[0m\33[49m\33[39m')

if __name__ == "__main__":
    printi("test")
    printw("test")
    printe("test")