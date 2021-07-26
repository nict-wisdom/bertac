# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
use strict;
use warnings;

if($#ARGV != 2){
	print "perl merge_masked.pl [MNLI|MRPC|QNLI|QQP|RTE|STS-B] [train|dev|dev_matched|dev_mismatched] base_dir\n";
	exit;
}

my $task = shift @ARGV;
my $stem = shift @ARGV;
my $base_dir = shift @ARGV;
unless($stem eq "train" or $stem eq "dev" or
	   $stem eq "dev_matched" or $stem eq "dev_mismatched"){ 
	print "perl merge_masked.pl [MNLI|MRPC|QNLI|QQP|RTE|STS-B] [train|dev|dev_matched|dev_mismatched] base_dir\n";
	exit;
}
unless($task eq "MNLI" or $task eq "MRPC" or
	   $task eq "QNLI" or $task eq "QQP" or
	   $task eq "RTE" or $task eq "STS-B" ){
	print "perl merge_masked.pl [MNLI|MRPC|QNLI|QQP|RTE|STS-B] [train|dev|dev_matched|dev_mismatched] base_dir\n";
	exit;
}

my $ifname1 = "$base_dir/glue_preprocess/NPMasking/masked/$task.$stem.masked";
my %db = ();

open(FIN, $ifname1) or die "Can't open $ifname1\n";
while(my $aline = <FIN>){
	chomp($aline);
	my @temp = split(/\t/,$aline);
	unless($#temp == 7 or $#temp == 5){
		print "$#temp: $aline\n";
		print "$task, $stem: EXIT\n";
		exit;
	}
	my $type = $temp[0];
	my $id = $temp[1];
	my $s1 = $temp[2];
	my $s2 = $temp[3];
	my $m1 = $temp[4];
	my $m2 = $temp[5];
	my $m3 = "null";
	my $m4 = "null";
	if($#temp > 5){
		$m3 = $temp[6];
		$m4 = $temp[7];
	}
	if($type eq "LCS"){
		$db{$id} = "$s1\t$s2\t$m1\t$m2\t$m3\t$m4";
	}
}
close FIN;

MNLI($stem, $base_dir) if($task eq "MNLI");
MRPC($stem, $base_dir) if($task eq "MRPC");
QNLI($stem, $base_dir) if($task eq "QNLI");
QQP($stem, $base_dir) if($task eq "QQP");
RTE($stem, $base_dir) if($task eq "RTE");
STSB($stem, $base_dir) if($task eq "STS-B");


sub MNLI{
	my ($stem, $bdir) = @_;
	my $ifname2 = "$bdir/glue_preprocess/download/MNLI/$stem.tsv";
	my $ofname = ">$bdir/GLUE/MNLI/$stem.tsv";
	my $line = 0;
	open(FIN, $ifname2) or die "Can't open $ifname2\n";
	open(FOUT, "$ofname") or die "Can't open $ofname\n";
	while(my $aline = <FIN>){
		chomp($aline);
		if($line != 0){
			my @temp = split(/\t/,$aline);
			my $idx  = $temp[0];
			my $sent1 = $temp[8];
			my $sent2 = $temp[9];
			my $tag   = $temp[$#temp];
			push @temp, $sent1;
			push @temp, $sent2;
			$aline = join("\t",@temp);
			if(exists($db{$idx})){
				my @masked = split(/\t/,$db{$idx});
				$temp[$#temp-1] = $masked[2];
				$temp[$#temp] = $masked[3];
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

sub MRPC{
	my ($stem, $bdir) = @_;
	my $ifname2 = "$bdir/glue_preprocess/download/MRPC/$stem.tsv";
	my $ofname = ">$bdir/GLUE/MRPC/$stem.tsv";
	my $line = 0;
	open(FIN, $ifname2) or die "Can't open $ifname2\n";
	open(FOUT, "$ofname") or die "Can't open $ofname\n";
	while(my $aline = <FIN>){
		chomp($aline);
		if($line != 0){
			my @temp = split(/\t/,$aline);
			my $tag = $temp[0];
			my $sid1 = $temp[1];
			my $sid2 = $temp[2];
			my $sent1 = $temp[3];
			my $sent2 = $temp[4];
			push @temp, $sent1;
			push @temp, $sent2;
			$aline = join("\t",@temp);
			if(exists($db{$line})){
				my @masked = split(/\t/,$db{$line});
				$temp[$#temp-1] = $masked[2];
				$temp[$#temp] = $masked[3];
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

sub QNLI{
	my ($stem, $bdir) = @_;
	my $ifname2 = "$bdir/glue_preprocess/download/QNLI/$stem.tsv";
	my $ofname = ">$bdir/GLUE/QNLI/$stem.tsv";
	my $line = 0;
	open(FIN, $ifname2) or die "Can't open $ifname2\n";
	open(FOUT, "$ofname") or die "Can't open $ofname\n";
	while(my $aline = <FIN>){
		chomp($aline);
		if($line != 0){
			my @temp = split(/\t/,$aline);
			my $idx  = $temp[0];
			my $sent1 = $temp[1];
			my $sent2 = $temp[2];
			push @temp, $sent1;
			push @temp, $sent2;
			$aline = join("\t",@temp);
			if(exists($db{$idx})){
				my @masked = split(/\t/,$db{$idx});
				$temp[$#temp-1] = $masked[2];
				$temp[$#temp] = $masked[3];
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

sub QQP{
	my ($stem, $bdir) = @_;
	my $ifname2 = "$bdir/glue_preprocess/download/QQP/$stem.tsv";
	my $ofname = ">$bdir/GLUE/QQP/$stem.tsv";
	my $line = 0;
	open(FIN, $ifname2) or die "Can't open $ifname2\n";
	open(FOUT, "$ofname") or die "Can't open $ofname\n";
	while(my $aline = <FIN>){
		chomp($aline);
		if($line != 0){
			my @temp = split(/\t/,$aline);
			unless($#temp < 5){
				my $idx  = $temp[0];
				my $sent1 = $temp[3];
				my $sent2 = $temp[4];
				my $tag   = $temp[5];
				push @temp, $sent1;
				push @temp, $sent2;
				$aline = join("\t",@temp);
				if(exists($db{$idx})){
					my @masked = split(/\t/,$db{$idx});
					$temp[$#temp-1] = $masked[2];
					$temp[$#temp] = $masked[3];
					my $new_sample = join("\t",@temp);
					print FOUT "$new_sample\n";
				}
				else{
					print FOUT "$aline\n";
				}
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

sub RTE{
	my ($stem, $bdir) = @_;
	my $ifname2 = "$bdir/glue_preprocess/download/RTE/$stem.tsv";
	my $ofname = ">$bdir/GLUE/RTE/$stem.tsv";
	my $line = -1;
	open(FIN, $ifname2) or die "Can't open $ifname2\n";
	open(FOUT, "$ofname") or die "Can't open $ofname\n";
	while(my $aline = <FIN>){
		chomp($aline);
		if($line != -1){
			my @temp = split(/\t/,$aline);
			my $idx = $temp[0];
			my $sent1 = $temp[1];
			my $sent2 = $temp[2];
			push @temp, $sent1;
			push @temp, $sent2;
			$aline = join("\t",@temp);
			if(exists($db{$idx})){
				my @masked = split(/\t/,$db{$idx});
				$temp[$#temp-1]= $masked[2];	
				$temp[$#temp]= $masked[3];	
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

sub STSB{
	my ($stem, $bdir) = @_;
	my $ifname2 = "$bdir/glue_preprocess/download/STS-B/$stem.tsv";
	my $ofname = ">$bdir/GLUE/STS-B/$stem.tsv";
	my $line = -1;
	open(FIN, $ifname2) or die "Can't open $ifname2\n";
	open(FOUT, "$ofname") or die "Can't open $ofname\n";
	while(my $aline = <FIN>){
		chomp($aline);
		if($line != -1){
			my @temp = split(/\t/,$aline);
			my $idx   = $temp[0];
			my $sent1 = $temp[7];
			my $sent2 = $temp[8];
			my $tag   = $temp[9];
			push @temp, 0 if $stem eq "test";
			push @temp, $sent1;
			push @temp, $sent2;
			$aline = join("\t",@temp);
			if(exists($db{$idx})){
				my @masked = split(/\t/,$db{$idx});
				$temp[$#temp-1]= $masked[2];	
				$temp[$#temp]= $masked[3];	
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
