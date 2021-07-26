# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.

use strict;
use warnings;

## POS TAGGING with spaCy
if($#ARGV != 3){
	print "perl np_extractor.pl set_stem[searchqa|quasart] script_dir tmp_dir num_parallel_processes\n";
	exit;
}
my $set = shift @ARGV;
my $pdir = shift @ARGV;
my $dir = shift @ARGV;
my $nparallel = shift @ARGV;

unless($set eq "searchqa" or $set eq "quasart"){
	print "perl np_extractor.pl set_stem[searchqa|quasart] script_dir tmp_dir num_parallel_processes\n";
	print "set_stem should be searchqa or quasart\n";
	exit;
}
my $idir = "$dir/sents";
my $odir = "$dir/np";

unless(-e $pdir){
	print "script_dir ($pdir) is not existing\n";
	exit;
}

unless(-e $idir){
	print "input data dir ($idir) is not existing\n";
	exit;

}
unless(-e $odir){
	print "output dir ($odir) is not existing\n";
	exit;
}

opendir(DH,$idir) or die "Can't open the dir $idir\n";
my @dirs = sort readdir(DH);
closedir DH;


my @cmd_set = ();
for(my $i = 0; $i <= $#dirs; $i++){
	if($dirs[$i] =~ /^($set.(?:train|dev|test))\.([0-9]+)/){
		my $stem1 = $1;
		my $stem2 = $2;
		my $cmd  = "python $pdir/spaCy_np_extractor.py --input_file $idir/$stem1.$stem2 > $odir/$stem1.$stem2.np &";
		push @cmd_set, $cmd;
		if($#cmd_set >= ($nparallel-1)){
			my $cmd_str = join(" ",@cmd_set);
			print "$cmd_str\n";
			system(qq{$cmd_str wait});
			@cmd_set = ();
		}
	}
}
if($#cmd_set != -1){
	my $cmd_str = join(" ",@cmd_set);
	print "$cmd_str\n";
	system(qq{$cmd_str wait});
}
