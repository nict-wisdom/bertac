# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
use strict;
use warnings;

if($#ARGV != 2){
	print "perl merge_masked.pl [CoLA|SST-2] [train|dev] base_dir\n";
	exit;
}

my $task = shift @ARGV;
my $stem = shift @ARGV;
my $base_dir = shift @ARGV;
unless($stem eq "train" or $stem eq "dev"){
	print "perl merge_masked.pl [CoLA|SST-2] [train|dev] base_dir\n";
	exit;
}
unless($task eq "CoLA" or $task eq "SST-2"){
	print "perl merge_masked.pl [CoLA|SST-2] [train|dev] base_dir\n";
	exit;
}

my $ifname1 = "$base_dir/glue_preprocess/NPMasking/masked/$task.$stem.masked";
my %db = ();

open(FIN, $ifname1) or die "Can't open $ifname1\n";
while(my $aline = <FIN>){
	chomp($aline);
	my @temp = split(/\t/,$aline);
	unless($#temp == 3){
		print "$#temp: $aline\n";
		exit;
	}
	my $type = $temp[0];
	my $id = $temp[1];
	my $s1 = $temp[2];
	my $m1 = $temp[3];
	if($type eq "LCS"){
		$db{$id} = "$s1\t$m1";
	}
}
close FIN;

if($task eq "CoLA"){
	CoLA($stem, $base_dir);
}
if($task eq "SST-2"){
	SST2($stem, $base_dir);
}

sub CoLA{
	my ($stem, $bdir) = @_;
	my $ifname2 = "$bdir/glue_preprocess/download/CoLA/$stem.tsv";
	my $ofname = ">$bdir/GLUE/CoLA/$stem.tsv";
	my $line = 0;
	open(FIN, $ifname2) or die "Can't open $ifname2\n";
	open(FOUT, "$ofname") or die "Can't open $ofname\n";
	while(my $aline = <FIN>){
		chomp($aline);
			my @temp = split(/\t/,$aline);
			my $tag   = $temp[1];
			my $sent1 = $temp[3];
			push @temp, $sent1;
			$aline = join("\t",@temp);
			my $idx   = $line;
			if(exists($db{$idx})){
				my @masked = split(/\t/,$db{$idx});
				$temp[$#temp] = $masked[1];
				my $new_sample = join("\t",@temp);
				print FOUT "$new_sample\n";
			}
			else{
				print FOUT "$aline\n";
			}
		$line++;
	}
	close FIN;
	close FOUT;
}
sub SST2{
	my ($stem, $bdir) = @_;
	my $ifname2 = "$bdir/glue_preprocess/download/SST-2/$stem.tsv";
	my $ofname = ">$bdir/GLUE/SST-2/$stem.tsv";
	my $line = 0;
	open(FIN, $ifname2) or die "Can't open $ifname2\n";
	open(FOUT, "$ofname") or die "Can't open $ofname\n";
	while(my $aline = <FIN>){
		chomp($aline);
		if($line != 0){
			my @temp = split(/\t/,$aline);
			my $sent1 = $temp[0];
			my $tag   = $temp[1];
			push @temp, $sent1;
			$aline = join("\t",@temp);
			my $idx   = $line;
			if(exists($db{$idx})){
				my @masked = split(/\t/,$db{$idx});
				$temp[$#temp] = $masked[1];
				my $new_sample = join("\t",@temp);
				print FOUT "$new_sample\n";
			}
			else{
				print FOUT "$aline\n";
			}
		}
		else{
			print FOUT "$aline\n";
		}
		$line++;
	}
	close FIN;
	close FOUT;
}
