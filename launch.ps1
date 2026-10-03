$cmd = 'cmd.exe /c ""d:\AIs\FaceKeyAnimation Process\RUN_TEST_STUDIO.bat"""'
$p = [wmiclass]"Win32_Process"
$res = $p.Create($cmd)
Write-Host "Launch Result Code: $($res.ReturnValue), ProcessId: $($res.ProcessId)"
