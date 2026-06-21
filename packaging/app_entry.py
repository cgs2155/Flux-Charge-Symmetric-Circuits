"""PyInstaller entry point: import the package (so relative imports work) and
launch the GUI.  Build with:  pyinstaller packaging/fluxcharge-gui.spec
"""
from fluxcharge.gui import main

if __name__ == "__main__":
    main()
