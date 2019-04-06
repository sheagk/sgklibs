#!/usr/bin/python
from __future__ import division, print_function


import h5py
from struct import pack
import sys,os
import numpy as np
from glob import glob
from argparse import ArgumentParser

def convert_ptype(inbase, outbase, ptype='PartType1', overwrite=False, verbose=False):
    files = glob(inbase+'*')
    files.sort()

    ptype_index = int(ptype[-1])

    if len(files) == 0:
        raise OSError("No files found starting with {}".format(inbase))

    solo = len(files) == 1
    for fname in files:
        if solo:
            outname = outbase
        else:
            outname = outbase + '/' + fname.split('/')[-1]
            if outname.endswith('.hdf5'):
                outname = outname[:-5]
            if outname.endswith('.h5'):
                outname = outname[:-3]
                
        if os.path.isfile(outname) and not overwrite:
            print("{0} already exists; skipping {1}".format(outname,fname))
            continue
                
        print("Dumping {} particles from {} to {}".format(ptype, fname,outname))

        with h5py.File(fname, 'r') as f, open(outname, 'wb') as out:
            fhead = f['Header'].attrs
            nfile = fhead['NumPart_ThisFile'][:]
            ntot = nfile[ptype_index]
            masstable = fhead['MassTable'][:]
            part = f[ptype]

            #start with the header:
            packedheader = b""
            ar = np.zeros(6, dtype='I')
            ar[ptype_index] = fhead['NumPart_ThisFile'][ptype_index]

            packedheader = packedheader + ar.tostring()

            mtable = np.zeros(6, dtype='d')
            if mtable[ptype_index] == 0:
                # check if we can just fix the mass table
                if (part['Masses'][:] == part['Masses'][0]).all():
                    mtable[1] = part['Masses'][0]
                    nmass = 0
                # otherwise write the masses
                else:
                    nmass = part['Masses'].shape[0]
                    
            packedheader = packedheader + mtable.tostring()

            ar = np.array(fhead['Time'],dtype='d')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['Redshift'],dtype='d')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['Flag_Sfr'],dtype='i')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['Flag_Feedback'],dtype='i')
            packedheader = packedheader + ar.tostring()

            ar = np.zeros(6, dtype='i')
            ar[ptype_index] = fhead['NumPart_Total'][ptype_index]
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['Flag_Cooling'],dtype='i')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['NumFilesPerSnapshot'],dtype='i')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['BoxSize'],dtype='d')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['Omega0'],dtype='d')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['OmegaLambda'],dtype='d')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['HubbleParam'],dtype='d')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['Flag_StellarAge'],dtype='i')
            packedheader = packedheader + ar.tostring()

            ar = np.array(fhead['Flag_Metals'],dtype='i')
            packedheader = packedheader + ar.tostring()

            ar = np.zeros(6, dtype='i')
            try:
                ar[ptype_index] = fhead['NumPart_Total_HW'][ptype_index]
            except KeyError:
                ar[ptype_index] = fhead['NumPart_Total_HighWord'][ptype_index]
            packedheader = packedheader + ar.tostring()

            if 'Flag_Entropy_ICs' in list(fhead.keys()):
                try:        #This is an array in at least one file that I have, so attempt to do it that way, and if it fails, do it as a float
                    ar = np.array(fhead['Flag_Entropy_ICs'][:],dtype='i')
                    packedheader = packedheader + ar.tostring()
                except TypeError:
                    ar = np.array(fhead['Flag_Entropy_ICs'],dtype='i')
                    packedheader = packedheader + ar.tostring()
            else:
                # print("Using Flag_IC_Info instead of Flag_Entropy_ICs.")
                ar = np.array(fhead['Flag_IC_Info'],dtype='i')
                packedheader = packedheader + ar.tostring()


            header_bytes_left = 256 - len(packedheader)
            for i in range(header_bytes_left):
                packedheader = packedheader + pack('<x')

            #now to write it out into a binary file
            out.write(pack('<I',256))
            out.write(packedheader)
            out.write(pack('<I',256))

            #Now to do coordinates, order of gas halo disk bulge star boundary
            vec_size = np.array([12*ntot],dtype='I')
            
            out.write(vec_size.tostring())
            ar = np.array(part['Coordinates'][:],dtype='f')
            out.write(ar.tostring())
            out.write(vec_size.tostring())

            if verbose: print("Finished with coordinates")

            #Now for velocities
            out.write(vec_size.tostring())
            ar = np.array(part['Velocities'][:],dtype='f')
            out.write(ar.tostring())
            out.write(vec_size.tostring())

            if verbose: print("Finished with velocities")

            #Now for particle IDs:
            float_size = np.array([4*ntot],dtype='I')
            
            out.write(float_size.tostring())
            ar = np.array(part['ParticleIDs'][:],dtype='I')
            out.write(ar.tostring())
            out.write(float_size.tostring())

            if verbose: print("Finished with particle IDs")

            #Now I have to check if there are variable particle masses
            if nmass > 0:
                nmass_size = np.array([4*nmass],dtype='I')
                out.write(nmass_size.tostring())
                ar = np.array(part['Masses'][:],dtype='f')
                out.write(ar.tostring())
                out.write(nmass_size.tostring())
                if verbose: print("Done writing masses for {0} particles".format(nmass))
        
        print("Finished with "+fname)

    return 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('inbase', help="Path (when appended with *) to the files to convert.  e.g. output/snapdir_600/snapshot")
    parser.add_argument('outbase', help="""Output file path.  Will be the exact name if only 
        one file; otherwise, should be a directory and file name will be made programatically.  e.g. gadget-output/snapdir_600/""")
    parser.add_argument('--ptype', help="Particle Type to convert.", default='PartType1')
    parser.add_argument('--overwrite', help="Overwrite existing files.", default=False, action='store_true')
    parser.add_argument('--verbose', help="print when we finish each particle.", default=False, action='store_true')

    args = parser.parse_args()
    convert_ptype(**args.__dict__)


    
