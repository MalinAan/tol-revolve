top -b | awk -v logfile=mem-proc.log '
{
        if($1 == "PID")
        {
                command="date +%T";
                command | getline ts
                close(command);
        }
        if($12 == "gzserver" || $12 == "body-analyzer")
        {
                printf "%s,%s,%s,%s,%s\n",ts,$1,$9,$10,$12 > logfile
        }
}'
