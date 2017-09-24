#!/bin/sh

# Simple script to run all programs provided to it and report whether
# they succeed or fail

LOG=test_results.txt
STDERR=test_stderr.txt

rm -f $LOG
touch $LOG

echo
echo "Writing output of test programs to $LOG"
echo

FAILURES=0

for TEST in "$@"
do
    if [ -x "$TEST" ]
    then
	rm -f $STDERR
	echo >> $LOG
	echo "########################################################" >> $LOG
	echo "### $TEST" >> $LOG
	echo "########################################################" >> $LOG
	echo >> $LOG
	echo -n "$TEST... "
	./$TEST >> $LOG 2> $STDERR
	if [ "$?" = 0 ]
	then
	    echo "PASSED"
	else
	    echo "*** FAILED ***"
	    cat $STDERR
	    FAILURES=`expr $FAILURES + 1`
	fi
    else
	echo "$TEST does not exist"
    fi
done

echo
if [ "$FAILURES" -gt "0" ]
then
    echo "$FAILURES programs failed in some way - see detailed output in $LOG"
else
    echo "All test programs ran successfully"
fi
echo

exit $FAILURES
