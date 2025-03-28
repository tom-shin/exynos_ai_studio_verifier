adb wait-for-device
adb root
adb remount

adb push lib64 /vendor/
adb push bin /vendor/

adb shell reboot
pause
