# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.

use strict;
use warnings;

## find a candidate for masked NP
if($#ARGV != 3){
	print "perl run_findcand.pl set_stem[searchqa|quasart] script_dir tmp_dir num_parallel_processes\n";
	exit;
}
my $set = shift @ARGV;
my $pdir = shift @ARGV;
my $tmp_dir = shift @ARGV;
my $nparallel = shift @ARGV;

unless($set eq "searchqa" or $set eq "quasart"){
	print "perl run_findcand.pl set_stem[searchqa|quasart] script_dir tmp_dir num_parallel_processes\n";
	print "set_stem should be searchqa or quasart\n";
	exit;
}
my $idir = "$tmp_dir/np";
my $odir = "$tmp_dir/cand";

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

opendir(DH,$idir) or die "Can't open the file\n";
my @dirs = sort readdir(DH);
closedir DH;


my @cmd_set = ();
for(my $i = 0; $i <= $#dirs; $i++){
	if($dirs[$i] =~ /^($set.(?:train|dev|test))\.([0-9]+)/){

		my $stem1 = $1;
		my $stem2 = $2;
		my $cmd  = "perl $pdir/find_npm_cand.pl $stem1.$stem2 $tmp_dir &";
		push @cmd_set, $cmd;
		if($#cmd_set > ($nparallel-1)){
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
