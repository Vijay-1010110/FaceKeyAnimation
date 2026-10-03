$cmd = 'cmd.exe /c ""d:\AIs\FaceKeyAnimation Process\RUN_LIVE_YOUTUBE.bat"""'
$p = [wmiclass]"Win32_Process"
$res = $p.Create($cmd)
Write-Host "Launch Result Code: $($res.ReturnValue), ProcessId: $($res.ProcessId)"
