# tools/create_shortcut.py
# Creates Windows .lnk shortcuts that point to your BAT launcher with a custom icon.
# Handles usernames with spaces and OneDrive Desktop folders.
# Requires: pip install pywin32

import os
import ctypes
import ctypes.wintypes
from pathlib import Path
from win32com.client import Dispatch


def get_desktop_path() -> Path:
    """
    Get the actual Desktop path even if it's synced with OneDrive or renamed.
    """
    CSIDL_DESKTOPDIRECTORY = 0x10
    SHGFP_TYPE_CURRENT = 0
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(
        0, CSIDL_DESKTOPDIRECTORY, 0, SHGFP_TYPE_CURRENT, buf
    )
    return Path(buf.value)


def get_start_menu_programs_path() -> Path:
    """
    Get the Start Menu\Programs folder for the current user.
    """
    return Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def create_shortcut(
    target_path: Path,
    shortcut_path: Path,
    icon_path: Path = None,
    working_dir: Path = None,
    description: str = "",
) -> None:
    """
    Creates a Windows .lnk shortcut.
    """
    target_path = target_path.resolve()
    shortcut_path = shortcut_path.resolve(strict=False)

    if working_dir is None:
        working_dir = target_path.parent

    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")

    if icon_path is not None and not icon_path.exists():
        raise FileNotFoundError(f"Icon not found: {icon_path}")

    # Ensure the shortcut directory exists
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)

    # Create shortcut using Windows Shell
    shell = Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(str(shortcut_path))

    # Wrap all paths with quotes to handle spaces safely
    shortcut.TargetPath = f'"{target_path}"'
    shortcut.WorkingDirectory = f'"{working_dir}"'
    shortcut.Description = description or shortcut_path.stem

    if icon_path is not None:
        shortcut.IconLocation = f'"{icon_path}",0'

    shortcut.WindowStyle = 1  # Normal window
    shortcut.Save()

    print(f"✅ Shortcut created: {shortcut_path}")


if __name__ == "__main__":
    # --- Adjust if your paths differ ---
    repo_root = Path(__file__).resolve().parents[1]  # ..\ai-research
    bat_launcher = repo_root / "PneumoNet.bat"       # Your .bat launcher name
    icon_file = repo_root / "assets" / "icons" / "lungai.ico"

    # Get target directories
    desktop_shortcut = get_desktop_path() / "LungAI Pneumonia Detection.lnk"
    start_menu_shortcut = get_start_menu_programs_path() / "LungAI Pneumonia Detection.lnk"

    description = "LungAI - Pneumonia Detection System (FastAPI + Vue)"

    # --- Create Desktop Shortcut ---
    try:
        create_shortcut(
            target_path=bat_launcher,
            shortcut_path=desktop_shortcut,
            icon_path=icon_file,
            working_dir=repo_root,
            description=description,
        )
    except Exception as e:
        print(f"❌ Failed to create Desktop shortcut: {e}")

    # --- Create Start Menu Shortcut (optional) ---
    try:
        create_shortcut(
            target_path=bat_launcher,
            shortcut_path=start_menu_shortcut,
            icon_path=icon_file,
            working_dir=repo_root,
            description=description,
        )
    except Exception as e:
        print(f"⚠️ Could not create Start Menu shortcut: {e}")
