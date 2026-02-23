$ErrorActionPreference = 'Stop'
function Emit-Active([string]$Err) {
  $payload = @{
    schema='ajax.human_signal.v1';
    ok=$false;
    last_input_age_sec=0;
    session_unlocked=$true;
    error=$Err;
  }
  $payload | ConvertTo-Json -Compress
}
try {
  Emit-Active 'stub_fail_closed'
} catch {
  Emit-Active ('stub_exception:' + $_.Exception.Message)
}
