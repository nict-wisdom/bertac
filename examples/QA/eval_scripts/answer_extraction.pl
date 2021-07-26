# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.

use strict;
use warnings;
use utf8;
use JSON;

use Encode;
use IO::Handle;
STDERR->autoflush(1);
STDOUT->autoflush(1);

my $debug = 1;

binmode STDOUT, ":encoding(utf8)";
binmode STDIN,  ":encoding(utf8)";
binmode STDERR, ":encoding(utf8)";

if($#ARGV != 4){
	print "perl answer_extraction.pl stem[quasart|searchqa] model_base[../model] selector_dir[albert-xxlarge-v2.quasart.1,2,3,4.100.TIER3.1e-05.e2] reader_dir[albert-xxlarge-v2.quasart.1,2,3,4.100.TIER3.1e-05.e2] stem[test|dev]\n";
	exit;
}

my $dstem = shift @ARGV;
unless($dstem eq "quasart" or $dstem eq "searchqa"){
	print "perl answer_extraction.pl stem[quasart|searchqa] model_base[../model] selector_dir[albert-xxlarge-v2.quasart.1,2,3,4.100.TIER3.1e-05.e2] reader_dir[albert-xxlarge-v2.quasart.1,2,3,4.100.TIER3.1e-05.e2] stem[test|dev]\n";
	print "dstem: $dstem\n";
	exit;
}
my $mstem = shift @ARGV;
my $sstem = shift @ARGV;
my $rstem = shift @ARGV;
my $dt_stem = shift @ARGV;
unless($dt_stem eq "dev" or $dt_stem eq "test"){
	print "perl answer_extraction.pl stem[quasart|searchqa] model_base[../model] selector_dir[albert-xxlarge-v2.quasart.1,2,3,4.100.TIER3.1e-05.e2] reader_dir[albert-xxlarge-v2.quasart.1,2,3,4.100.TIER3.1e-05.e2] stem[test|dev]\n";
	print "stem: $dt_stem\n";
	exit;
}

my @temp = split(/\./,$rstem);
my @temp2 = split(/\./,$sstem);
if($temp[1] ne $dstem){
	print "dstem and reader stem should have the same prefix\n";
	print "DSTEM: $dstem\n";
	print "RSTEM_PREFIX: $temp[0].$temp[1]\n";
	exit;
}
if($temp2[1] ne $dstem){
	print "dstem and selector stem should have the same prefix\n";
	print "DSTEM: $dstem\n";
	print "SSTEM_PREFIX: $temp2[0].$temp2[1]\n";
	exit;
}


my $base = $mstem;

my $rdir = "$base/reader";
my $sdir = "$base/selector";

my %sdb = ();
read_selector($sdir, $sstem);

sub read_selector{
	my ($cdir, $cmodel) = @_;
	my $fname = "$cdir/$cmodel/predictions_$dt_stem.json";
	my $tag = $cmodel;
	print "Load selector: $fname\n";
	open(FIN,"<:encoding(utf8)",$fname) or die "Can't open the file $fname\n";
	while(my $aline = <FIN>){
		chomp($aline);
		my @temp = split(/[ \t]/,$aline);
		my $qid = $temp[0];
		my $pid = $temp[1];
		my $score = $temp[2];
		$sdb{"$qid $pid"} = $score;
	}
	close FIN;
}

my %rdb = ();
my $rfname = "$rdir/$rstem/3best_predictions_$dt_stem.txt";

print "Load reader: $rfname\n";
open(FIN,"<:encoding(utf8)",$rfname) or die "Can't open [$rfname]\n";
while(my $aline = <FIN>){
	chomp($aline);
	my @temp = split(/\t/,$aline);
	my $id = $temp[0];
	my $pid = $temp[1];
	my $text = $temp[2];
	my $score = $temp[3];
	$rdb{$id}{$pid} = "$text\t$score";
}
close FIN;

my %tprob = ();
my @ids = sort keys %rdb;
print "Load tprob\n";
for(my $i =0; $i <= $#ids; $i++){
	my @all_ids = split(/ /,$ids[$i]);
	my $qid = $all_ids[0];
	if(!exists($sdb{$ids[$i]})){
		print "Can't find selector prob for $ids[$i]\n";
		exit;
	}
	my $p_prob = $sdb{$ids[$i]};
	my @alist = sort keys %{$rdb{$ids[$i]}};
	for(my $j =0; $j <= $#alist; $j++){ # rank
		my @temp = split(/\t/,$rdb{$ids[$i]}{$alist[$j]});
		my $text = normalize_answer($temp[0]);
	    my $prob = $temp[1];
		$tprob{$qid}{$text} += $p_prob * $prob;
	}
}

my $odir = "$dt_stem";

my $cur_dir = ".";
my $ofname = "$cur_dir/extracted_answers/$odir/$dstem/$sstem.$rstem.out";
print "Write json files to $ofname\n";
open(FOUT,">:encoding(utf8)",$ofname) or die "Can't open the output $ofname\n";
my @res = sort {$a<=>$b} keys %tprob;
for(my $k =0; $k <= $#res; $k++){
	my @ans = sort {$tprob{$res[$k]}{$b}<=>$tprob{$res[$k]}{$a}} keys %{$tprob{$res[$k]}};
	my $cans = $ans[0];
	if($ans[0] eq ""){
		if($#ans >= 1){
			$cans = $ans[1];
		}
	}
	print FOUT encode_json({ ids => $res[$k], answer => $cans, score=>$tprob{$res[$k]}{$cans} }) ."\n";
}
close FOUT;

sub normalize_answer{
	my ($ans) = @_;
	# do lower
	$ans =~ tr/A-Z/a-z/;
	$ans = "" if $ans eq "no_answer" or $ans eq "empty";
	# remove punc
	$ans =~ s/[[:punct:]]//g;
	# remove articles
	$ans =~ s/ (?:a|an|the) / /g;
	$ans =~ s/^(?:a|an|the) //g;
	$ans =~ s/ (?:a|an|the)$//g;
	# white space fix
	$ans =~ s/[ ]+/ /g;
	$ans =~ s/^[ ]+//g;
	$ans =~ s/[ ]+$//g;
	return $ans;
}
