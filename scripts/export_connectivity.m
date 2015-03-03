#! /usr/bin/octave -qf
args = argv ();

m = load(args{1});

GCtoGC   = m.connection_M{1,1};
MCtoGC   = m.connection_M{2,1};
HCtoGC   = m.connection_M{3,1};
BCtoGC   = m.connection_M{4,1};
AACtoGC  = m.connection_M{5,1};
HCCtoGC  = m.connection_M{6,1};
NGFCtoGC = m.connection_M{7,1};

GCtoMC  = m.connection_M{1,2};
MCtoMC  = m.connection_M{2,2};
HCtoMC  = m.connection_M{3,2};
BCtoMC  = m.connection_M{4,2};
AACtoMC = m.connection_M{5,2};
HCCtoMC = m.connection_M{6,2};

GCtoHC  = m.connection_M{1,3};
MCtoHC  = m.connection_M{2,3};
HCtoHC  = m.connection_M{3,3};
BCtoHC  = m.connection_M{4,3};

GCtoBC   = m.connection_M{1,4};
MCtoBC   = m.connection_M{2,4};
HCtoBC   = m.connection_M{3,4};
BCtoBC   = m.connection_M{4,4};
HCCtoBC  = m.connection_M{6,4};
NGFCtoBC = m.connection_M{7,4};

GCtoAAC   = m.connection_M{1,5};
MCtoAAC   = m.connection_M{2,5};
HCtoAAC   = m.connection_M{3,5};
NGFCtoAAC = m.connection_M{7,5};

GCtoHCC   = m.connection_M{1,6};
MCtoHCC   = m.connection_M{2,6};
HCCtoHCC  = m.connection_M{6,6};
NGFCtoHCC = m.connection_M{7,6};

HCtoNGFC   = m.connection_M{3,7};
NGFCtoNGFC = m.connection_M{7,7};

fid = fopen ('MCtoGC.dat','w+');
fprintf(fid, '%d %d\n', size(MCtoGC)(1), size(MCtoGC)(2));
fclose(fid);
save('-ascii', '-append', 'MCtoGC.dat', 'MCtoGC');

