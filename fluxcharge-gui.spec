# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for the fluxcharge desktop app.
#
#   pip install pyinstaller
#   pyinstaller fluxcharge-gui.spec --noconfirm     # run from the repo root
#
# Produces dist/fluxcharge-gui.app (macOS) / dist/fluxcharge-gui/ (Win/Linux).
# Building a bundled app (rather than running the script from a terminal) makes
# it a proper foreground GUI app, so mouse clicks are delivered reliably.
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
hiddenimports += collect_submodules('fluxcharge')
tmp_ret = collect_all('schemdraw')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['packaging/app_entry.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fluxcharge-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='fluxcharge-gui',
)
app = BUNDLE(
    coll,
    name='fluxcharge-gui.app',
    icon=None,
    bundle_identifier='org.fluxcharge.gui',
    info_plist={
        'CFBundleName': 'fluxcharge',
        'CFBundleDisplayName': 'fluxcharge',
        'NSHighResolutionCapable': True,
        # a normal foreground app (gets focus and mouse events reliably)
        'LSUIElement': False,
    },
)
