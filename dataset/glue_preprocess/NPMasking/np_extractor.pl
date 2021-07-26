# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
use strict;
use warnings;

if($#ARGV != 0){
	print "perl np_extractor.pl base_dir\n";
	exit;
}
my $base_dir = shift @ARGV;

my @all_tasks = ("CoLA" ,"MNLI" ,"MRPC" ,"QNLI", "QQP" ,"RTE" ,"SST-2" ,"STS-B");

my @cmd_set = ();
for(my $a = 0; $a <= $#all_tasks; $a++){
	my $task = $all_tasks[$a];
	my @dset = ("train","dev");
	@dset = ("train", "dev_matched", "dev_mismatched") if $task eq "MNLI";
	for(my $i =0; $i <= $#dset; $i++){
		my $cmd = "python spaCy_np_extractor.py --input $base_dir/download/$task/$dset[$i].tsv --task $task > np/$task.$dset[$i].tag &";
		push @cmd_set, $cmd;
	}
}

# simply do the parallel processing for all the eight tasks 
my $cmd_str = join(" ",@cmd_set);
system(qq{$cmd_str wait});

