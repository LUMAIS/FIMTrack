#!/bin/sh
# Generate Arguments parser

sh -c 'mkdir Autogen'  2> /dev/null
gengetopt  --output-dir Autogen -i args.ggo

if [ $? -eq 0 ]; then
	echo  "The arguments parser is generated"
else
	echo  "The arguments parser generation is FAILED"
fi
