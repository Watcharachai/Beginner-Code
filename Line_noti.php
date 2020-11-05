<?php
    $command = escapeshellcmd('python3 /Users/rodtour/Desktop/Line_notify.py');
    $output = shell_exec($command);
    echo $output;
?>