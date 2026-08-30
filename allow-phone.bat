@echo off
rem Let a phone on the same network reach this app.
rem
rem Windows blocks incoming connections to new programs, and it does it
rem silently: the app binds its port, prints an address, draws a QR code, and
rem the phone then sits there forever. Nothing in the application can fix
rem that, because changing firewall rules needs administrator rights.
rem
rem So this asks for them. Double-click it, say yes to the prompt, and it adds
rem one rule scoped to your own network.

setlocal
set "PORT=8000"
set "RULE=TextSearchVDO"

rem Already elevated? net session only succeeds as administrator.
net session >nul 2>&1
if %errorlevel% equ 0 goto :elevated

echo.
echo   This needs administrator rights to change a firewall rule.
echo   Windows will ask - say yes.
echo.
rem Re-launch this same file elevated. -Verb RunAs is what raises the prompt.
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
exit /b 0

:elevated
echo.
echo   Allowing phones on your local network to reach TextSearchVDO
echo   on port %PORT%.
echo.

powershell -NoProfile -Command ^
  "Get-NetFirewallRule -DisplayName '%RULE%' -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue"

rem -RemoteAddress LocalSubnet keeps this to your own network rather than
rem opening the port outright. This only ever needs to serve a device sitting
rem on the same WiFi, hotspot or cable.
powershell -NoProfile -Command ^
  "New-NetFirewallRule -DisplayName '%RULE%' -Direction Inbound -Protocol TCP -LocalPort %PORT% -Action Allow -Profile Any -RemoteAddress LocalSubnet | Out-Null"

if errorlevel 1 (
    echo   Could not add the rule.
    echo.
    pause
    exit /b 1
)

echo   Done. Checking it took effect...
echo.
powershell -NoProfile -Command ^
  "$r = Get-NetFirewallPortFilter | Where-Object { $_.LocalPort -contains '%PORT%' } | Get-NetFirewallRule | Where-Object { $_.Direction -eq 'Inbound' -and $_.Enabled -eq 'True' -and $_.Action -eq 'Allow' }; if ($r) { Write-Host '   Allowed. Reload the page on your phone.' } else { Write-Host '   The rule is not there. Something removed it.' }"

echo.
echo   To undo this later:
echo     Remove-NetFirewallRule -DisplayName "%RULE%"
echo.
pause
