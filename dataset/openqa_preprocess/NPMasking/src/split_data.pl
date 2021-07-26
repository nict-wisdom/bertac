# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
use strict;
use warnings;

#
# Split data into smaller parts for parallel processing
#

if($#ARGV != 2){
	print "perl split_data.pl set[searchqa|quasart] stem[train|dev|test] output_dir\n";
	exit;
}

my $set_stem = shift @ARGV;
my $stem = shift @ARGV;
my $odir = shift @ARGV;

unless($set_stem eq "searchqa" or $set_stem eq "quasart"){
	print "perl split_data.pl set[searchqa|quasart] stem[train|dev|test] input_data_dir output_dir\n";
	exit;
}

unless($stem eq "train" or $stem eq "dev" or $stem eq "test"){
	print "perl split_data.pl set[searchqa|quasart] stem[train|dev|test] input_data_dir output_dir\n";
	exit;
}

my $idir = $set_stem;
unless(-e $idir){
	print "input_data dir ($idir) is not existing\n";
	exit;
}
unless(-e $odir){
	print "output dir ($odir) is not existing\n";
	exit;
}

my $idx = 0;
my %vocab = ();
open(FIN,"$idir/$stem.json.txt") or die "Can't open the file $idir/$stem.json.txt\n";
open(FOUT,">$odir/$set_stem.$stem.$idx") or die "Can't open the file\n";
my $line = 0;
while(my $aline = <FIN>){
	chomp($aline);
	$line++;
	if($line % 100000 == 0){
		close FOUT;
		$idx++;
		open(FOUT,">$odir/$set_stem.$stem.$idx") or die "Can't open the file\n";

	}
	print FOUT "$aline\n";
	my @temp = split(/\t/,$aline);
	my @temp2 = split(/[ ]+/,$temp[1]);
	for(my $i =0; $i <= $#temp2; $i++){
		$vocab{$temp2[$i]} = 1;
	}
}
close FIN;
close FOUT;


open(FOUT,">$odir/$set_stem.$stem.vocab.txt") ;
my @key = sort keys %vocab;
for(my $i =0; $i <= $#key; $i++){
	print FOUT "$key[$i]\n";
}
close FOUT;
