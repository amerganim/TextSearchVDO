' Double-click launcher that opens no console window at all.
'
' TextSearchVDO.bat still works and is the one to use when something is
' wrong, because it can print. But a .bat is run by cmd.exe, and cmd.exe
' shows a window - so double-clicking it flashes a black console before the
' app appears, no matter that it goes on to call pythonw.exe. Windows has no
' way to suppress that from inside the batch file itself.
'
' This has no console to begin with. WScript.Shell.Run with a window style of
' 0 starts pythonw.exe hidden, and anything that goes wrong before the window
' exists is reported in a message box rather than vanishing.

Option Explicit

Dim shell, fso, here, pythonw, marker
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = here

pythonw = fso.BuildPath(here, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonw) Then
    MsgBox _
        "TextSearchVDO is not set up yet." & vbCrLf & vbCrLf & _
        "Run this once, from this folder:" & vbCrLf & vbCrLf & _
        "    py -3.14 -m venv .venv" & vbCrLf & _
        "    .venv\Scripts\python -m pip install -r requirements.txt" & vbCrLf & _
        "    .venv\Scripts\python -m tsv setup", _
        vbExclamation, "TextSearchVDO"
    WScript.Quit 1
End If

' A missing detector is not fatal - motion search still works - so this warns
' and carries on rather than refusing to start.
marker = fso.BuildPath(here, "data\models")
If fso.FolderExists(marker) Then
    If Not fso.FileExists(fso.BuildPath(marker, "yolo11n.onnx")) And _
       Not fso.FileExists(fso.BuildPath(marker, "yolox_tiny.onnx")) And _
       Not fso.FileExists(fso.BuildPath(marker, "yolox_s.onnx")) Then
        MsgBox _
            "No object detection model is installed." & vbCrLf & vbCrLf & _
            "The app will open and can still find movement, but it cannot " & _
            "recognise people or search by description." & vbCrLf & vbCrLf & _
            "To add one:  .venv\Scripts\python -m tsv setup", _
            vbInformation, "TextSearchVDO"
    End If
Else
    MsgBox _
        "No models are installed yet." & vbCrLf & vbCrLf & _
        "Run:  .venv\Scripts\python -m tsv setup", _
        vbExclamation, "TextSearchVDO"
End If

' 0 = hidden window, False = do not wait for it to finish.
shell.Run """" & pythonw & """ -m tsv app", 0, False
