#!/usr/bin/env python
"""
WinML Setup Checker
Verifies WASDK Python bindings and Windows App SDK runtime are installed.
"""

import subprocess
import sys
import re


def friendly_version_key(friendly_version):
    """Return a sortable key for marketing version strings."""
    if not friendly_version:
        return (-1,)

    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?(?:-(.+))?", friendly_version)
    if not match:
        return (-1,)

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or 0)
    suffix = match.group(4)

    # Stable builds (no suffix) should rank above previews of the same major/minor.
    suffix_priority = 0 if not suffix else -1
    suffix_number = 0
    if suffix:
        digits = re.search(r"(\d+)$", suffix)
        suffix_number = int(digits.group(1)) if digits else 0

    return (major, minor, patch, suffix_priority, suffix_number)


def get_pip_package_version(package_name):
    """Get installed pip package version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def list_windows_app_sdk_packages():
    """Return a list of installed Windows App SDK runtimes."""
    packages = []
    seen = set()
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-AppxPackage -Name 'Microsoft.WindowsAppRuntime.*' | "
             "Where-Object { $_.Name -notmatch \'\\.(Main|Singleton|DDLM|CBS)' } | "
             "Sort-Object Version | "
             "ForEach-Object { $_.Name + '|' + $_.Version }"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if '|' not in line:
                    continue
                name, internal = line.strip().split('|', 1)
                internal = internal.strip()
                # Older naming carries the marketing version in the package name,
                # e.g. 'Microsoft.WindowsAppRuntime.2.0-experimental6' or '...1.8'.
                match = re.search(r'WindowsAppRuntime\.(\d+\.\d+(?:\.\d+)?(?:-[\w\d]+)?)', name)
                if match:
                    friendly = match.group(1)
                else:
                    # Newer stable/preview packages use a single-major name like
                    # 'Microsoft.WindowsAppRuntime.2', '...2-preview2', '...2-experimental7'.
                    # The real marketing version lives in the internal version field
                    # (e.g. 2.2.0.0 -> 2.2.0), so derive the friendly version from it.
                    name_match = re.search(r'WindowsAppRuntime\.(\d+)(?:-([A-Za-z]+\d*))?$', name)
                    if name_match:
                        channel = name_match.group(2)
                        ver_match = re.match(r'(\d+\.\d+\.\d+)', internal)
                        base = ver_match.group(1) if ver_match else internal
                        friendly = f"{base}-{channel}" if channel else base
                    else:
                        friendly = None
                key_id = (friendly, internal)
                if key_id in seen:
                    continue
                seen.add(key_id)
                packages.append({
                    'name': name.strip(),
                    'friendly': friendly,
                    'internal': internal.strip(),
                    'key': friendly_version_key(friendly)
                })
    except Exception:
        return []
    return packages


def parse_wasdk_version(version_str):
    """Parse wasdk version string like '2.0.0.dev4' into SDK version info."""
    if not version_str:
        return None

    # Dev versions like 2.0.0.dev4 -> 2.0.0-experimental4
    match = re.match(r'(\d+)\.(\d+)\.(\d+)\.dev(\d+)', version_str)
    if match:
        major, minor, patch, dev = match.groups()
        return {
            'sdk_version': f"{major}.{minor}.{patch}-experimental{dev}",
            'download_url': f"https://aka.ms/windowsappsdk/{major}.{minor}/"
                           f"{major}.{minor}.{patch}-experimental{dev}/"
                           f"windowsappruntimeinstall-x64.exe",
            'is_preview': True
        }

    # Newer experimental packages use a semantic patch like '2.1.3'.
    # Keep the full version (small patch = real semver, not a build number).
    # Older stable packages use a large build number like '1.8.260209005',
    # which should be truncated to 'major.minor'.
    match = re.match(r'(\d+)\.(\d+)\.(\d+)$', version_str)
    if match:
        major, minor, patch = match.groups()
        if int(patch) < 1000:
            return {
                'sdk_version': f"{major}.{minor}.{patch}",
                'download_url': "https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads",
                'is_preview': True
            }
        return {
            'sdk_version': f"{major}.{minor}",
            'download_url': "https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads",
            'is_preview': False
        }
    return None


def print_status(label, ok, details=""):
    """Print formatted status line."""
    print(f"  [{'✓' if ok else '✗'}] {label}: {details}")


def main():
    print("=" * 60)
    print("WinML Setup Checker")
    print("=" * 60)
    print(f"\nPython: {sys.version.split()[0]} ({sys.executable})\n")

    # Check WASDK Python packages
    print("WASDK Python Packages:")
    print("-" * 40)

    ml_ver = get_pip_package_version("wasdk-Microsoft.Windows.AI.MachineLearning")
    boot_ver = get_pip_package_version(
        "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap")

    if ml_ver:
        print_status("wasdk-ML", True, ml_ver)
    else:
        print_status("wasdk-ML", False, "NOT INSTALLED")
        print("    pip install --pre wasdk-Microsoft.Windows.AI.MachineLearning[all]")

    if boot_ver:
        print_status("wasdk-Bootstrap", True, boot_ver)
    else:
        print_status("wasdk-Bootstrap", False, "NOT INSTALLED")
        print("    pip install --pre wasdk-Microsoft.Windows.ApplicationModel."
              "DynamicDependency.Bootstrap")

    if ml_ver and boot_ver and ml_ver != boot_ver:
        print(f"\n  [!] WARNING: Version mismatch! ML: {ml_ver} vs Bootstrap: {boot_ver}")

    # Check Windows App SDK Runtime
    print("\nWindows App SDK Runtime:")
    print("-" * 40)

    packages = list_windows_app_sdk_packages()
    best_pkg = max(packages, key=lambda p: p['key']) if packages else None
    friendly = best_pkg['friendly'] if best_pkg else None
    internal = best_pkg['internal'] if best_pkg else None
    wasdk_info = parse_wasdk_version(ml_ver)

    if internal:
        display = f"{friendly} (internal: {internal})" if friendly else internal
        print_status("Windows App SDK", True, display)
    else:
        print_status("Windows App SDK", False, "NOT INSTALLED")
        print("    Windows App SDK Runtime is required but not installed.")
        if wasdk_info:
            print(f"    Required version: {wasdk_info['sdk_version']}")
            print(f"    Download from: {wasdk_info['download_url']}")
        else:
            print("    Download from: https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads")

    if packages:
        print("\nInstalled runtimes (newest first):")
        for pkg in sorted(packages, key=lambda p: p['key'], reverse=True):
            label = pkg['friendly'] or "Unknown marketing version"
            marker = "*" if best_pkg and pkg is best_pkg else "-"
            print(f"    {marker} {label} (internal: {pkg['internal']})")
        if best_pkg:
            print("    * Active runtime used by this checker")

    # Recommendations
    if wasdk_info and internal:
        print(f"\nwasdk package version : {wasdk_info['sdk_version']} (pip)")
        print(f"Active runtime version: {friendly} (internal: {internal})")
        print( "Note: All stable 2.x runtimes share the 'WindowsAppRuntime.2' package family.")
        print( "      A newer runtime (e.g. 2.2.0) supersedes older ones (e.g. 2.1.3) in-place")
        print( "      and is fully compatible with the wasdk pip package.")

    # Summary
    print("\n" + "=" * 60)
    sdk_ok = internal is not None
    all_ok = ml_ver and boot_ver and sdk_ok
    if all_ok:
        print("Status: All components installed. Please, ensure matching Windows App SDK version is Installed.")
    else:
        missing = []
        if not ml_ver:
            missing.append("wasdk-ML package")
        if not boot_ver:
            missing.append("wasdk-Bootstrap package")
        if not sdk_ok:
            missing.append("Windows App SDK Runtime")
        print(f"Status: Setup incomplete. Missing: {', '.join(missing)}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
