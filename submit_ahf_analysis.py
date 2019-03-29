#!/usr/bin/env python3

import os, sys
from subprocess import check_call, check_output, Popen
from glob import glob

EMAIL_ADDRESS = 'dummy@liamg.com'


# if this is true, then the sbatch files won't actually be submitted
dryrun = False

# if this is true, then existing output files will be overwritten
# this has no effect on consistent_trees -- it'll be run regardless
overwrite = False

def find_run_name(path):
    possibilities = path.split('/')
    for pos in possibilities:
        if 'elvis' in pos:
            return pos.strip('m12_elvis_')
        if 'm09' in pos or 'm10' in pos or 'm11' in pos or 'm12' in pos:
            return pos
    return path

#### options for the jobs that we'll actually submit 
run_name              = find_run_name(os.getcwd())

# number of jobs to run at once and number per node for AHF halo finding:
ahf_nnodes            = 10
ahf_ntasks_per_node   = 4
ahf_ompthreads        = 8
ahf_jobname           = 'ahf-'+run_name
ahf_ntasks            = ahf_nnodes * ahf_ntasks_per_node

# number of jobs to run at once and number for node for building the AHF trees:
mtree_ntasks          = 96
mtree_ntasks_per_node = 24
mtree_jobname         = 'mtree-'+run_name

ctrees_jobname        = 'ctree-'+run_name

# sbatch options that are in common across all setps
sbatch_options = {
                'mail-user':EMAIL_ADDRESS,  # will do begin, end, and fail
                'partition':'cca',
                'time':'168:00:00',
                'exclusive':'',
                }


#### file names and paths:
simulation_directory = '../..'
snapshot_inbase      = simulation_directory+'/output/snapdir_'
home                 = os.path.expanduser('~')+'/code/halo'

## input parameter files:
# file that defines all the AHF inputs (other than fname, outname, file type)
ahf_param_base       = home+'/AHF.input'
ctree_param_base     = home+'/ctree-input.cfg'

ahf_jobfile          = 'ahf_commands.sh'
ahf_sbatchfile       = 'run_ahf.slurm'

mtree_jobfile        = 'mtree_commands.sh'
mtree_sbatchfile     = 'run_mtree.slurm'

ctree_sbatchfile     = 'run_ctrees.slurm'

### output file paths:
# overall output directory
catalog_directory = './catalog'

# directory for MergerTree outputs
mtree_directory            = catalog_directory + '/ahf-trees'

# directories for consistent_trees outputs
hlist_directory            = catalog_directory + '/hlists'
tree_directory             = catalog_directory + '/trees'
tree_outputs_directory     = catalog_directory + '/outputs'

# directory for disBatch log files:
disbatch_log_directory     = 'disbatch-logs'

#### paths that hopefully shouldn't need to be changed:
python          = 'python3'
perl            = 'perl'

# path to execuatables
ahf_exec        = home + '/ahf-v1.0-094/bin/AHF-v1.0-094'
mtree_exec      = home + '/ahf-v1.0-094/bin/MergerTree'

ctree_directory = home + '/consistent-trees'
tree_script     = "do_merger_tree_np.pl"   # or remove the np for periodic boundaries
hlist_script    = "halo_trees_to_catalog.pl"

## path to scripts to convert hdf5->gadget and AHF_halos + mtree_idx->out_xxx.list
temp_gbin_dir   = 'temp-gbins'
hdf5_to_gbin    = home + '/convert_ptype_to_gbin.py'
this_script     = os.path.realpath(__file__)

# directories for AHF
ahf_halo_directory         = catalog_directory + '/halos'
ahf_input_directory        = catalog_directory + '/inputs'
ahf_log_directory          = catalog_directory + '/logs'
ahf_parameter_directory    = catalog_directory + '/paramters'
ahf_particle_directory     = catalog_directory + '/particles'
ahf_profile_directory      = catalog_directory + '/profiles'
ahf_substructure_directory = catalog_directory + '/substructure'
ahf_std_directory          = catalog_directory + '/std-logs'
#####################




### define some useful functions
def checkdir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

def write_slurm(f, options, job_name, ntasks, ntasks_per_node, 
    disbatch_command_file=None, disbatch_log_base=None, 
    pre_disbatch_commands=[], post_disbatch_commands=[]):
    def write_sbatch_line(key, val=''):
        string = '#SBATCH --'+key
        if len(val):        
            string += '='+val
        f.write(string+'\n')

    f.write('#!/usr/bin/env bash\n')
    f.write('\n')
    write_sbatch_line('job-name', job_name)
    write_sbatch_line('ntasks', str(ntasks))
    write_sbatch_line('ntasks-per-node', str(ntasks_per_node))
    for key, val in options.items():
        write_sbatch_line(key, val)
    
    if 'mail-user' in options:
        write_sbatch_line('mail-type', 'begin')
        write_sbatch_line('mail-type', 'end')
        write_sbatch_line('mail-type', 'fail')

    f.write('\n')

    # set the umask -- files created should be readable for group and other
    f.write('umask 0022\n\n')

    # write any pre disBatch commands (e.g. export OMP_NUM_THREADS)
    for cmd in pre_disbatch_commands:
        f.write('{}\n'.format(cmd))

    # call disBatch:
    if disbatch_command_file is not None:
        # make sure we've loaded the module we need
        f.write('module load disBatch\n')

        if disbatch_log_base is not None:
            f.write('disBatch.py -p {} {}\n\n'.format(disbatch_log_base, disbatch_command_file))
        else:
            f.write('disBatch.py {}\n\n'.format(disbatch_log_base, disbatch_command_file))

    # write any post disBatch commands (e.g. moving files or calling do_ctrees or )
    for cmd in post_disbatch_commands:
        f.write('{}\n'.format(cmd))

    # couple blank lines at the end for safety
    f.write('\n\n')
    return

## make sure all our definitions are ok
# don't want to output in current directory
assert catalog_directory != ""

# stuff we need for AHF to work
assert os.path.isfile(ahf_param_base)
assert os.path.isfile(hdf5_to_gbin) 
assert os.path.isfile(ahf_exec) 

# stuff we need for MergerTree to work
assert os.path.isfile(mtree_exec)

# stuff we need for consistent trees to work
assert os.path.isdir(ctree_directory)
assert os.path.isfile(ctree_directory+'/'+tree_script)
assert os.path.isfile(ctree_directory+'/'+hlist_script)
assert os.path.isfile(ctree_param_base)

all_output_directories = [temp_gbin_dir, catalog_directory, 
    ahf_halo_directory, ahf_input_directory, ahf_log_directory, 
    ahf_parameter_directory, ahf_substructure_directory, disbatch_log_directory,
    ahf_particle_directory, ahf_profile_directory, ahf_std_directory, 
    mtree_directory, hlist_directory, tree_directory, tree_outputs_directory]

# make sure the output directories exist
for directory in all_output_directories:
    checkdir(directory)

def run_ahf():
    '''
    run ahf using the paths defined above

    creates the .AHF_input files based on an input parameter file
    then creates a list of commands for disBatch.py to handle
    then creates a SLURM file to run those commands, and submits it
    '''

    snapshots = glob(snapshot_inbase+"*")
    if len(snapshots) == 0:
        raise OSError("No snapshots found!  Exiting")
    snapshots.sort()

    ##### now halo find on all the snapshots using disBatch
    # read in the lines in the paramter file
    f = open(ahf_param_base,'r')
    paramlines = [line for line in f]
    f.close()

    print("Setting up halo finding on {} snapshots".format(len(snapshots)))
    ## loop over the snapshots, create the input files, and create the commands to run
    commands = []
    for ii in range(len(snapshots)):
        snap = snapshots[ii]
        
        snapnum = snap.split('_')[-1]
        if '.' in snapnum:
            snapnum = snapnum.split('.')[0]
        snapnum = int(snapnum)

        if os.path.isdir(snap):
            conversion_inbase = snap + '/'
            doing_dir = True
            sname = glob(conversion_inbase+'/*')[0].split('/')[-1].split('.')[0]
            ahf_filetype = '61'
        else:
            conversion_inbase = snap
            doing_dir = False
            sname = snap.split('/')[-1].rsplit('.', 1)[0]
            ahf_filetype = '60'

        #build the AHF.input file -- only ic_filename, ic_filetype, and outfile_prefix get edited
        ahf_inbase = temp_gbin_dir + '/' + sname
        outfile_exists = len(glob(catalog_directory + '/' + sname + '*AHF_halos')) or len(glob(ahf_halo_directory+ '/' + sname + '*AHF_halos'))
        if outfile_exists and (overwrite==False):   #output file exists and we don't want  to overwrite
            print("Already have a halo file for {0}; skipping it.".format(sname))
            continue

        if doing_dir and not ahf_inbase.endswith('.'):
            ahf_inbase += '.'

        paramf = ahf_input_directory+'/{0}'.format(sname)
        with open(paramf,'w') as param:
            for l in paramlines:
                if l.startswith('ic_filename'):
                    l = 'ic_filename        = '+ahf_inbase+'\n'
                elif l.startswith('outfile_prefix'):
                    l = 'outfile_prefix     = '+catalog_directory + '/' + sname + '\n'
                elif l.startswith('ic_filetype'):
                    l = 'ic_filetype        = '+ahf_filetype+'\n'
                param.write(l)

        logfile = ahf_std_directory+'/'+sname+'.stdout'

        # convert, run, clean up
        commands.append(r'{} {} {} {} && {} {} &> {} && rm {}*'.format(
            python, hdf5_to_gbin, conversion_inbase, temp_gbin_dir,
            ahf_exec, paramf, logfile,
            ahf_inbase))

    # now dump the commands to a file:
    with open(ahf_jobfile, 'w') as jobf:
        jobf.write('\n'.join(commands))

    # create the sbatch file to do the actual work:
    if ahf_ompthreads > 0:
        pre_disbatch_commands = ['export OMP_NUM_THREADS='+str(ahf_ompthreads)]
    else:
        pre_disbatch_commands = []

    post_disbatch_commands = [
        'mv ' + catalog_directory + '/*.AHF_halos '        + ahf_halo_directory         + '/',
        'mv ' + catalog_directory + '/*.log '              + ahf_log_directory          + '/',
        'mv ' + catalog_directory + '/*.parameter '        + ahf_parameter_directory    + '/',
        'mv ' + catalog_directory + '/*.AHF_profiles '     + ahf_profile_directory      + '/',
        'mv ' + catalog_directory + '/*.AHF_particles '    + ahf_particle_directory     + '/',
        'mv ' + catalog_directory + '/*.AHF_substructure ' + ahf_substructure_directory + '/',
        python+' '+this_script+' 2']

    with open(ahf_sbatchfile, 'w') as sbatchf:
        write_slurm(sbatchf, sbatch_options, ahf_jobname, ahf_ntasks, ahf_ntasks_per_node,
            disbatch_command_file=ahf_jobfile, disbatch_log_base=disbatch_log_directory+'/ahf',
            pre_disbatch_commands=pre_disbatch_commands, 
            post_disbatch_commands=post_disbatch_commands)

    print("Submitting {} to do the halo finding ({} tasks at once with {} per node)".format(
        ahf_jobfile, ahf_ntasks, ahf_ntasks_per_node))

    if dryrun is not True:
        job_info = check_output(['sbatch', ahf_sbatchfile])
        job_id = int(job_info.strip().split()[-1])

        print("When job {} finishes, it should resubmit this script ({}) with an argument of 2".format(job_id, this_script))
    else:
        print("Actually, you'll have to do the submission of {} yourself...".format(ahf_jobfile))
        return -1


def run_mtree():
    '''
    run MergerTree using the paths defined above.  AHF must have been run first

    creates a list of commands for disBatch.py to handle that are MergerTree < 2 "input file ii" "input file ii+1" "output file"
    then creates a SLURM file to run those commands, and submits it
    '''

    # lambda function to get snapshot name from the AHF particle files
    snapname = lambda fname:  fname.split('/')[-1].split('.z')[0]

    particle_files = glob(ahf_particle_directory+'/*particles')
    particle_files.sort()

    if not len(particle_files):
        raise OSError("Didn't find any particle files in {}".format(ahf_particle_directory))

    commands = []
    for ii in range(len(particle_files)-1):
        file1 = particle_files[ii]
        file2 = particle_files[ii+1]
        outfile = mtree_directory + '/' + snapname(file1)+'_to_'+snapname(file2)
        if os.path.isfile(outfile+'_mtree_idx') and (overwrite==False):
            print("Already have {}, so skipping {} to {}".format(
                outfile, snapname(file1), snapname(file2)))
            continue
        commands.append(mtree_exec + ' <<< "2 '+file1+' '+file2+' '+outfile+'"')

    with open(mtree_jobfile, 'w') as jobf:
        jobf.write('\n'.join(commands))

    # no pre commands necessary here; post is to rerun this script with arg 3
    post_disbatch_commands = [python+' '+this_script+' 3']
    with open(mtree_sbatchfile, 'w') as sbatchf:
        write_slurm(sbatchf, sbatch_options, mtree_jobname, mtree_ntasks, mtree_ntasks_per_node,
            disbatch_command_file=mtree_jobfile, disbatch_log_base=disbatch_log_directory+'/mtree',
            post_disbatch_commands=post_disbatch_commands)

    print("Submitting {} to run MergerTree ({} tasks at once with {} per node".format(
        mtree_jobfile, mtree_ntasks, mtree_ntasks_per_node))

    if dryrun is not True:
        job_info = check_output(['sbatch', mtree_sbatchfile])
        job_id = int(job_info.strip().split()[-1])
        print("When job {} finishes it should resubmit this script {} with an argument of 3".format(job_id, this_script))
        return job_id
    else:
        print("Actually, you'll have to do the submission of {} yourself...".format(mtree_sbatchfile))
        return -1


def run_ctrees():
    '''
    creats a job script that will 
        1. builds the out_xxx.lists (using this file)
        2. runs ctrees, 
        3. runs hlists
    '''
    import gizmo_analysis

    ### call this script to make the out.lists (no need to worry about variables cause they're set internally)
    commands = ['set -e']
    commands.append('export PYTHONPATH=$HOME/stock-wetzel/src')
    cmd = [python, this_script, 'out']
    commands.append(' '.join(cmd))

    ### need to load a snapshot for the cosomology + box size (though latter probably irrelevant for non-periodic boxes)
    header = gizmo_analysis.gizmo_io.Read.read_header(simulation_directory=simulation_directory)

    Om = header['omega_matter']
    Ol = header['omega_lambda']
    h0 = header['hubble']
    boxwidth = header['box.length/h']

    cwd = os.getcwd()+'/'

    #build the input file using absolute paths
    cfgname = cwd+"ctrees.cfg"
    with open(cfgname, 'w') as cfg, open(ctree_param_base, 'r') as incfg:
        #Write the relevant file paths
        cfg.write('SCALEFILE     = '+cwd+catalog_directory+'/DescScale.txt\n')
        cfg.write('INBASE        = '+cwd+catalog_directory+'\n')
        cfg.write('OUTBASE       = '+cwd+tree_outputs_directory+'\n')
        cfg.write('TREE_OUTBASE  = '+cwd+tree_directory+'\n')
        cfg.write('HLIST_OUTBASE = '+cwd+hlist_directory+'\n')
        cfg.write('\n')
        cfg.write('Om={} #Omega_Matter\n'.format(Om))
        cfg.write('Ol={} #Omega_Lambda\n'.format(Ol))
        cfg.write('h0={} #h0\n'.format(h0))
        cfg.write('BOX_WIDTH={} #h0\n'.format(boxwidth))

        #write the rest of the lines, making sure to skip the lines I already wrote
        for line in incfg:
            l = line.strip()
            if l.split('=')[0].strip() in ['SCALEFILE', 'INBASE', 'OUTBASE', 'TREE_OUTBASE', 'HLIST_OUTBASE', 'Om', 'Ol', 'h0']:
                continue
            cfg.write(line)

    #now cd to the ctrees directory and call the do_merger_tree.pl:
    #call is:  perl do_merger_tree_np.pl <consistent-trees directory> <consistent-trees config file>
    commands.append('cd '+ctree_directory)
    commands.append(' '.join([perl, tree_script, ctree_directory, cfgname]))

    # now make the hlists:
    #call is:  perl halo_trees_to_catalog.pl <consistent-trees config file>
    commands.append(' '.join([perl, hlist_script, cfgname]))

    with open(ctree_sbatchfile, 'w') as sbatchf:
        write_slurm(sbatchf, sbatch_options, ctrees_jobname, 1, 1, 
            pre_disbatch_commands=commands)

    print("Submitting {} to build the out_xxx.list files, make the tree_0.0.0.dat file, and create the hlist files".format(
        ctree_sbatchfile))

    if dryrun is not True:
        job_info = check_output(['sbatch', ctree_sbatchfile])
        job_id = int(job_info.strip().split()[-1])

        print("When job {} finishes, you should be able to create the HDF5 files with submit_rockstar_hdf5.py".format(job_id))
        return job_id
    else:
        print("Actually, you'll have to do the submission yourself...")
        return -1


def build_out_lists():
    import gizmo_analysis

    halofiles = glob(ahf_halo_directory+'/*_halos')
    halofiles.sort()

    idxfiles = glob(mtree_directory+'/*_mtree_idx')
    idxfiles.sort()

    assert len(halofiles) == len(idxfiles)+1

    # load a snapshot to get information for the header:  need Om, Ol, h, particle mass, box size
    part = gizmo_analysis.gizmo_io.Read.read_snapshots(
        species='dark', simulation_directory=simulation_directory, properties='mass')

    boxlength_o_h = part.info['box.length/h']/1e3
    Om = part.Cosmology['omega_matter']
    Ol = part.Cosmology['omega_lambda']
    h = part.Cosmology['hubble']
    particle_mass = np.median(part['dark']['mass']) * h
    del part

    def load_scalefactor(index):
        return gizmo_analysis.gizmo_io.Read.read_header(
            snapshot_value=index, simulation_directory=simulation_directory)['scalefactor']

    def make_header(colheads, scale):
        header = '' 
        header += colheads + '\n' 
        header += 'a = {:7.6f}\n'.format(scale)
        header += 'Om = {:7.6f}; Ol = {:7.6f}; h = {:7.6f}\n'.format(Om, Ol, h)
        header += 'Particle mass:  {:.5e} Msun/h\n'.format(particle_mass)
        header += 'Box size:  {:.6f} Mpc/h\n'.format(boxlength_o_h)
        header += 'file created from AHF halo catalogs'
        return header

    scalelist = []
    fnums = []

    #want:  #ID DescID Mass Vmax Vrms Radius Rs Np X Y Z VX VY VZ JX JY JZ Spin
    #add on:  Rmax, r2, sigV, cNFW
    #if baryons, also add on Mstar, Mgas
    #don't have:  Jx, Jy, Jz, but use Lx, Ly, Lz; no Vrms, but can be zeros
    with open(halofiles[-1],'r') as f:
        if 'gas' in f.readline():
            baryons = True
            print("Looks like you included baryons in the halo finding, so I'll carry Mgas etc. through")
        else:
            baryons = False
            print("Don't see evidence of baryons in the halo finding")

    # output columns (copied from Andrew's rockstar_io):
    column_heads ="ID   DescID   M200b(Msun/h)   Vmax   Vrms   R200b  Rs   Np   "
    column_heads+="X   Y   Z   Vx   Vy   Vz  Lx   Ly   Lz    "
    column_heads+="Spin   rs_klypin M200b_tot   Mvir    M200c   M500c   M100m   Xoff    Voff    "
    column_heads+="Spin_Bullock     b_to_a  c_to_a  A[x]    A[y]    A[z]    "
    column_heads+="b_to_a(500c)    c_to_a(500c) A[x](500c)  A[y](500c)  A[z](500c)  "
    column_heads+="T/|U|   M_pe_Behroozi   M_pe_diemer Type     "
    column_heads+="SM   Gas     BH_mass     m200b_highres   m200b_lowres"

    # don't have:
    # M200b_tot (set = M200b), Mvir, M200c, M500c, M100m, A[x], A[y], A[z], 
    # anything at 500c, M_pe_Behroozi, M_pe_diemer, Voff
    # Type, but that all seems to be 0 as far as I can tell

    # calculate:
    # rs_klypin from Rmax
    # T/U from T and U
    # m200b_highres and m200b_lowres from m200b and fMhires

    iddt = int

    for ii in range(len(halofiles)):
        fname = halofiles[ii].split('/')[-1]
        file_rshift = float(fname.split('z')[-1][:-len('.AHF_halos')])
        file_scale = 1./(1+file_rshift)

        snum = int(fname.split('_')[1].split('.z')[0])
        outname = catalog_directory+'/out_{0:03}.list'.format(snum)

        with open(halofiles[ii],'r') as f:
            lc = 0
            for line in f:
                lc += 1
                if lc > 2:
                    break

        if lc < 2 or (lc==2 and line==''):
            print("! no halos in {0}!".format(halofiles[ii]))
            continue    
            
        scale = load_scalefactor(snum)
        rshift = (1./scale) - 1
        assert (np.abs(rshift - file_rshift)/(0.5*(rshift+file_rshift)) < 0.01) \
                or (np.abs(scale - file_scale)/(0.5*(scale+file_scale)) < 0.01)

        # only save non-blank files to the DescScales file, so I don't try to run ctrees on them -- i.e. put this after the lc check
        # but save files that I've already done, so put this before the overwrite/existence check
        fnums.append(snum)
        scalelist.append(scale)     # use the higher precision scale factor from the snapshot file

        if path.isfile(outname) and (not overwrite):
            print("Already have {}; skipping it.".format(outname))
            continue

        #input columns, annoyingly starting at 1
        #   ID(1)  hostHalo(2)     numSubStruct(3) Mvir(4) npart(5)        
        #   Xc(6)   Yc(7)   Zc(8)   VXc(9)  VYc(10) VZc(11) Rvir(12)      
        #   Rmax(13)        r2(14)  mbp_offset(15)  com_offset(16)  
        #   Vmax(17)        v_esc(18)       sigV(19)        lambda(20)    
        #   lambdaE(21)     Lx(22)  Ly(23)  Lz(24)  b(25)   c(26)   
        #   Eax(27) Eay(28) Eaz(29) Ebx(30) Eby(31) Ebz(32) Ecx(33)       
        #   Ecy(34) Ecz(35) ovdens(36)      nbins(37)       fMhires(38)     
        #   Ekin(39)        Epot(40)        SurfP(41)    Phi0(42) cNFW(43)
        #   n_gas(44)       M_gas(45)       lambda_gas(46)  lambdaE_gas(47) 
        #   Lx_gas(48)      Ly_gas(49)      Lz_gas(50)      b_gas(51)       
        #   c_gas(52)       Eax_gas(53)     Eay_gas(54)     Eaz_gas(55)     
        #   Ebx_gas(56)     Eby_gas(57)     Ebz_gas(58)     Ecx_gas(59)     
        #   Ecy_gas(60)     Ecz_gas(61)     Ekin_gas(62)    Epot_gas(63)    
        #   n_star(64)      M_star(65)      lambda_star(66) lambdaE_star(67)        
        #   Lx_star(68)     Ly_star(69)     Lz_star(70)     b_star(71)      
        #   c_star(72)      Eax_star(73)    Eay_star(74)    Eaz_star(75)    
        #   Ebx_star(76)    Eby_star(77)    Ebz_star(78)    Ecx_star(79)    
        #   Ecy_star(80)    Ecz_star(81)    Ekin_star(82)   Epot_star(83)

        hal_prog_ids = np.loadtxt(halofiles[ii],unpack=True,usecols=0,dtype=iddt)
        if baryons:
            mass, vmax, veldisp, radius, rs, numpart, \
                x, y, z, vx, vy, vz, lx, ly, lz, \
                spin_peebles, rmax, xoff, \
                Spin_Bullock, b_to_a, c_to_a, \
                kinetic, potential, mstar, mgas, \
                fMhires = np.loadtxt(halofiles[ii], unpack=True,
                    usecols=[3, 16, 18, 11, 13, 4, 
                             5, 6, 7, 8, 9, 10, 21, 22, 23,
                             20, 12, 15, 
                             19, 24, 25, 
                             38, 39, 64, 44, 
                             37])
        else:
            mass, vmax, veldisp, radius, rs, numpart, \
                x, y, z, vx, vy, vz, lx, ly, lz, \
                spin_peebles, rmax, xoff, \
                Spin_Bullock, b_to_a, c_to_a, \
                kinetic, potential, \
                fMhires = np.loadtxt(halofiles[ii], unpack=True,
                    usecols=[3, 16, 18, 11, 13, 4, 
                             5, 6, 7, 8, 9, 10, 21, 22, 23,
                             20, 12, 15, 
                             19, 24, 25, 
                             38, 39, 
                             37])
            mstar = np.zeros(mass.size, dtype=int)
            mgas = np.zeros(mass.size, dtype=int)

        numpart = numpart.astype('int')

        x /= 1e3
        y /= 1e3
        z /= 1e3

        rs_klypin = rmax/2.1626

        T_over_U = kinetic / np.abs(potential)
        mhighres = fMhires * mass
        mlowres = (1.0 - fMhires) * mass

        # lots of columns I don't have in AHF unfortunately
        fill_array = np.empty(hal_prog_ids.size, dtype=int)
        fill_array.fill(-1)

        zero_array = np.zeros(hal_prog_ids.size, dtype=int)

        hal_desc_ids = np.empty_like(hal_prog_ids)     
        hal_desc_ids.fill(-1)

        if halofiles[ii] != halofiles[-1]:   # leave as -1 in the last timestep; no descendants then
            tree_prog_ids,tree_desc_ids = np.loadtxt(idxfiles[ii],unpack=True,dtype=iddt)

            # indices of the halos that have descendants in the tree -- every halo in the tree is in the catalogs, but not vice versa:
            hal_indices = np.where(np.isin(hal_prog_ids, tree_prog_ids))[0]

            # check that everything matches up:
            assert (hal_prog_ids[hal_indices] == tree_prog_ids).all()

            # now fill in the descendants (where I have them) using the trees
            hal_desc_ids[hal_indices] = tree_desc_ids

        header = make_header(column_heads, scale)
        output_data = np.column_stack((
            hal_prog_ids, hal_desc_ids, mass, vmax, veldisp, radius, rs, numpart, 
            x, y, z, vx, vy, vz, lx, ly, lz, 
            spin_peebles, rs_klypin, mass, fill_array, fill_array, fill_array, fill_array, xoff, fill_array,   #fills are the masses I don't have and voff
            Spin_Bullock, b_to_a, c_to_a, fill_array, fill_array, fill_array,   #fills are the allgood shape vectors
            fill_array, fill_array, fill_array, fill_array, fill_array,   # fills are shape etc at 500c
            T_over_U, fill_array, fill_array, zero_array,  #fills are smoothed masses, zeros are type
            mstar, mgas, zero_array, mhighres, mlowres))    #zeros are BH_mass

        # cast everything to a string before saving it
        np.savetxt(outname, output_data, header=header, comments='#', fmt='%s')  
        print("Wrote {0}".format(outname))

    # and finally save the scale factors of the halo catalogs
    np.savetxt(outdir+'DescScale.txt',np.column_stack((fnums,scalelist)),fmt="%d %5f")

#### steps are:
if __name__ == '__main__':
    '''
    this is a python script to automate the process of 
    running the amiga halo finder on a GIZMO/FIRE simulation, 
    then running the MergerTree utility in Amiga Halo Finder
    (which creates links between timesteps based on the
    particle IDs in the halos), then massage those outputs
    into a format that Consistent Trees can handle, then
    call consistent trees on those massaged outputs.

    everything except ctrees is automatically parallelized,
    since each timestep (or in the case of ctrees, each 
    timestep1 -> timestep2 link) is independent of 
    everything else.  set the options at the top of the 
    script to handle how much parallelization you want,
    and to point to the appropriate files.

    you'll need:
        * AHF, of course, compiled with whatever options you want
        * MergerTree
        * consistent-trees
        * diBatch, which handles the spreading out of the jobs
        * ahf_param_base -- a AHF.input file with the parameters you 
            want to use set.  input name, output name, and file
            types will be set for each snapshot, but all other 
            parameters will be kep the same throughout.
        * ctree_param_base -- a cfg file for consistent trees with 
            the options that you want to use.  as with ahf_param_base,
            filenames will be set automatically; cosmology and box 
            size will be as well.  all other options will be copied
            over though.
        * hdf5_to_gbin -- a script that converts the GIZMO HDF5 snapshots
            into gadget binary snapshots.  I believe this step can be
            removed, as the latest AHF versions have raw support for 
            HDF5 files.  However, take a look in the base repository 
            for an old and ugly script that can do this conversion
            for a single particle type (e.g. if you want to run AHF 
            on just dark matter, which is a pretty common use case)

    Once you have all your files in place and all your arguments set 
    appropriately at the top of this script, run it with 
        $ python run_ahf_analysis.py 1

    That'll create the AHF.input files and make a list of commands needed 
    to do the halo finding on all of the snapshots.  It'll then create and 
    call an sbatch file that uses disBatch to run that list of commands in 
    paralell.  Once disBatch is done (i.e. all the halo finding has been run),
    this script will call itself again with an argument of 2, i.e. the 
    equivalent of doing
        $ python run_ahf_analysis.py 2

    That'll do the same (create a list of commands and an sbatch file to run them
    via disBatch), but for the MergerTree step.  Again, once it'd done,  it'll 
    call this script again with an argument of 3:
        $ python run_ahf_analysis.py 3

    That step will create (then submit) an sbatch file that builds the out.lists
    then runs consistent trees, then you should be done!
    '''


    if len(sys.argv) != 2:
        raise OSError("Must call with a step of 1, 2, or 3 (or ahf, mtree, ctrees) to run the appropriate step")
    step_todo = sys.argv[1]

    # step 1 / ahf:  run AHF
    if step_todo == '1' or step_todo == 'ahf':
        run_ahf()

    # step 2 / mtree:  run MergerTree:
    elif step_todo == '2' or step_todo == 'mtree':
        run_mtree()

    # step 3 / ctrees:  run consistent_trees
    elif step_todo == '3' or step_todo == 'ctrees':
        run_ctrees()

    # step 3.5 / out:  build the out.lists (should be run on a compute node)
    elif step_todo == '3.5' or step_todo == 'out':
        build_out_lists()

    else:
        raise OSError("Don't know how to do step {}".format(step_todo))












