#!/usr/bin/env python3
"""
ORLIB Set Cover Benchmark Downloader - Enhanced Version

Downloads set cover instances from the ORLIB repository and converts them to
both MPS (Gurobi model) and pickle (data structure) formats.

Features:
- Parallel downloading for faster processing
- Retry logic for failed downloads
- Comprehensive error reporting and debugging
- Progress tracking with detailed statistics

Original code adapted from the GeCO repository:
https://github.com/CharJon/GeCO/blob/main/geco/mips/loading/orlib.py
"""

from gurobipy import Model, GRB
import gurobipy as gp
import os
import pickle
import argparse
import time
from urllib.request import urlopen, Request, HTTPRedirectHandler, HTTPHandler, HTTPSHandler, build_opener
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback

FILES_BASE_URL = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"

# Global lock for thread-safe printing
print_lock = Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe printing function."""
    with print_lock:
        print(*args, **kwargs)

def read_number(line):
    if not line:
        return None
    return _str_to_number(line.strip().split(b" ")[0])


def read_numbers(line):
    return (_str_to_number(n) for n in line.strip().split(b" "))


def read_multiline_numbers(file, number_to_read):
    numbers = []
    while file:
        if len(numbers) == number_to_read:
            break
        elif len(numbers) > number_to_read:
            raise ValueError("Found more numbers than expected")
        else:
            line = file.readline()
            numbers_in_line = list(read_numbers(line))
            numbers += numbers_in_line
    return numbers


def _str_to_number(string):
    if b"." in string:
        return float(string)
    else:
        return int(string)


def zero_index(lst):
    return [x - 1 for x in lst]


def orlib_load_instance(instance_name, reader, formulation, timeout=30):
    """
    Load instance with timeout and better error handling.
    
    Parameters
    ----------
    instance_name: str
        Name of instance file
    reader: function (file) -> params: tuple
        Takes a file-like object and returns the read parameters
    formulation: function (params: tuple) -> gurobi.model
        Takes a tuple of params and returns the generated model
    timeout: int
        Timeout in seconds for the download
        
    Returns
    -------
    model: gurobi.Model
        A Gurobi model of the loaded instance
    params: tuple
        The parsed instance parameters
    """
    try:
        import socket
        from io import BytesIO
        
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        
        try:
            # Try original URL first
            base_url = FILES_BASE_URL + instance_name
            
            # Create opener that handles redirects
            opener = build_opener(HTTPRedirectHandler, HTTPHandler, HTTPSHandler)
            
            try:
                response = opener.open(base_url)
                content = response.read()
            except Exception as e:
                # If original URL fails, try HTTPS version
                https_url = base_url.replace('http://', 'https://')
                try:
                    response = opener.open(https_url)
                    content = response.read()
                except Exception:
                    raise e  # Raise original error if HTTPS also fails
            
            # Create file-like object from content
            content_as_file = BytesIO(content)
            
            params = reader(content_as_file)
            model = formulation(*params)
            return model, params
            
        finally:
            socket.setdefaulttimeout(old_timeout)
            
    except (URLError, HTTPError) as e:
        raise Exception(f"Network error downloading {instance_name}: {e}")
    except Exception as e:
        raise Exception(f"Error processing {instance_name}: {e}")


def set_cover_gurobi(costs, sets): 
    """Create a Gurobi model for the set cover problem."""
    try:
        model = gp.Model('set_cover') 
        model.setParam('OutputFlag', 0) 
        variables = [model.addVar(vtype=GRB.BINARY, obj=c, name=f"v_{i}") for i, c in enumerate(costs)]
        model.update() 
        
        for i, s in enumerate(sets):
            if len(s) > 0:  # Skip empty sets
                model.addConstr(gp.quicksum(variables[j] for j in s) >= 1, name=f"set_constraint_{i}")
        
        model.modelSense = GRB.MINIMIZE
        model.update()
        return model
    except Exception as e:
        raise Exception(f"Error creating Gurobi model: {e}")


def scp_reader(file):
    """Reads scp set-cover instances with better error handling."""
    try:
        number_of_cons, number_of_vars = read_numbers(file.readline())
        costs = read_multiline_numbers(file, number_of_vars)
        sets = []
        
        for constraint_idx in range(number_of_cons):
            number_of_vars_in_constraint = read_number(file.readline())
            if number_of_vars_in_constraint is None:
                break
            if number_of_vars_in_constraint <= 0:
                sets.append([])  # Empty constraint
                continue
                
            constraint = list(read_multiline_numbers(file, number_of_vars_in_constraint))
            constraint = zero_index(constraint)
            # Validate constraint indices
            valid_constraint = [idx for idx in constraint if 0 <= idx < number_of_vars]
            if len(valid_constraint) != len(constraint):
                thread_safe_print(f"Warning: Invalid indices found in constraint {constraint_idx}")
            sets.append(valid_constraint)
        
        # Validation
        if len(costs) != number_of_vars:
            raise ValueError(f"Cost vector length {len(costs)} != {number_of_vars}")
        if len(sets) != number_of_cons:
            raise ValueError(f"Constraint count {len(sets)} != {number_of_cons}")
            
        return costs, sets
    except Exception as e:
        raise Exception(f"Error parsing SCP format: {e}")


def rail_reader(file):
    """Reads rail set-cover instances with better error handling."""
    try:
        number_of_cons, number_of_vars = read_numbers(file.readline())
        
        costs = []
        sets = [[] for _ in range(number_of_cons)]
        col_idx = 0
        
        while file:
            line = file.readline()
            if not line:
                break
                
            numbers = list(read_numbers(line))
            if len(numbers) < 2:
                continue
                
            costs.append(numbers[0])
            if len(numbers) > 2:
                rows_covered = zero_index(numbers[2:])
                # Validate row indices
                valid_rows = [row for row in rows_covered if 0 <= row < number_of_cons]
                if len(valid_rows) != len(rows_covered):
                    thread_safe_print(f"Warning: Invalid row indices in column {col_idx}")
                
                for row in valid_rows:
                    sets[row].append(col_idx)
            col_idx += 1
        
        # Filter empty sets
        sets = [s for s in sets if len(s) > 0]
        
        # Validation
        if len(costs) != number_of_vars:
            raise ValueError(f"Cost vector length {len(costs)} != {number_of_vars}")
            
        return costs, sets
    except Exception as e:
        raise Exception(f"Error parsing RAIL format: {e}")


def orlib_instance(instance_name, timeout=30):
    """Load an orlib Set-cover instance with timeout."""
    if instance_name[:3] == "scp":
        return orlib_load_instance(
            instance_name, reader=scp_reader, formulation=set_cover_gurobi, timeout=timeout
        )
    elif instance_name[:4] == "rail":
        return orlib_load_instance(
            instance_name, reader=rail_reader, formulation=set_cover_gurobi, timeout=timeout
        )
    else:
        raise ValueError(f"Unknown instance type for {instance_name}")


def download_single_instance(args):
    """Download a single instance - designed for parallel execution."""
    file_name, base_folder, skip_existing, timeout, max_retries = args
    
    result = {
        'name': file_name,
        'success': False,
        'skipped': False,
        'error': None,
        'num_variables': 0,
        'num_constraints': 0,
        'download_time': 0,
        'retry_count': 0
    }
    
    try:
        base_name = file_name.split('.')[0]
        mps_filename = os.path.join(base_folder, base_name + ".mps")
        data_filename = os.path.join(base_folder, base_name + ".pkl")
        
        # Check if files already exist
        if skip_existing and os.path.exists(mps_filename) and os.path.exists(data_filename):
            result['skipped'] = True
            result['success'] = True
            return result
        
        # Download with retries
        start_time = time.time()
        last_error = None
        
        for retry in range(max_retries + 1):
            try:
                result['retry_count'] = retry
                
                # Download and parse the instance
                gp_model, data = orlib_instance(file_name, timeout=timeout)
                costs, sets = data
                
                # Create instance data structure
                instance_data = {
                    'name': file_name,
                    'costs': costs, 
                    'sets': sets,
                    'num_variables': len(costs),
                    'num_constraints': len(sets),
                    'instance_type': file_name[:3] if file_name[:3] == 'scp' else 'rail'
                }
                
                # Save MPS file
                gp_model.write(mps_filename)
                
                # Save pickle file with instance data
                with open(data_filename, 'wb') as f:
                    pickle.dump(instance_data, f)
                
                result['success'] = True
                result['num_variables'] = len(costs)
                result['num_constraints'] = len(sets)
                result['download_time'] = time.time() - start_time
                break
                
            except Exception as e:
                last_error = str(e)
                if retry < max_retries:
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    result['error'] = last_error
                    break
        
        if not result['success']:
            result['error'] = last_error or "Unknown error"
            
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
    
    return result


def get_all_instance_names():
    """Get the complete list of ORLIB set cover instance names."""
    scp_file_names = []
    
    # Problem set 4: scp41 to scp410
    for i in range(41, 50):  # scp41 to scp49
        scp_file_names.append(f"scp{i}.txt")
    scp_file_names.append('scp410.txt')
    
    # Problem set 5: scp51 to scp510  
    for i in range(51, 60):  # scp51 to scp59
        scp_file_names.append(f"scp{i}.txt")
    scp_file_names.append('scp510.txt')
    
    # Problem set 6: scp61 to scp65
    for i in range(61, 66):  # scp61 to scp65
        scp_file_names.append(f"scp{i}.txt")
    
    # Problem sets A-E
    for i in range(1, 6):
        scp_file_names.extend([
            f"scpa{i}.txt", f"scpb{i}.txt", f"scpc{i}.txt", f"scpd{i}.txt", 
            f"scpe{i}.txt"
        ])
    
    # Problem sets E-H (NR variants)  
    for i in range(1, 6):
        scp_file_names.extend([
            f"scpnre{i}.txt", f"scpnrf{i}.txt", 
            f"scpnrg{i}.txt", f"scpnrh{i}.txt"
        ])
    
    # CLR and CYC unicost problems
    scp_file_names.extend([
        'scpclr10.txt', 'scpclr11.txt', 'scpclr12.txt', 'scpclr13.txt',
        'scpcyc06.txt', 'scpcyc07.txt', 'scpcyc08.txt', 'scpcyc09.txt', 
        'scpcyc10.txt', 'scpcyc11.txt'
    ])
    
    # Rail instances (first 3 smaller ones)
    scp_file_names.extend([
        'rail507.txt', 'rail516.txt', 'rail582.txt'
    ])
    
    return sorted(scp_file_names)


def download_instances_parallel(instance_names, base_folder, skip_existing=True, max_workers=4, timeout=30, max_retries=2):
    """
    Download and process multiple set cover instances in parallel.
    
    Parameters
    ----------
    instance_names: list[str]
        List of instance file names to download
    base_folder: str
        Directory to save files
    skip_existing: bool
        Whether to skip instances that already exist
    max_workers: int
        Maximum number of parallel download threads
    timeout: int
        Timeout in seconds for each download
    max_retries: int
        Maximum number of retries for failed downloads
        
    Returns
    -------
    dict: Summary of download results
    """
    os.makedirs(base_folder, exist_ok=True)
    
    print(f"Starting parallel download of {len(instance_names)} instances...")
    print(f"Workers: {max_workers}, Timeout: {timeout}s, Max retries: {max_retries}")
    print(f"Output directory: {base_folder}")
    print("=" * 70)
    
    start_time = time.time()
    results = []
    
    # Prepare arguments for parallel execution
    download_args = [
        (file_name, base_folder, skip_existing, timeout, max_retries)
        for file_name in instance_names
    ]
    
    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_instance = {
            executor.submit(download_single_instance, args): args[0] 
            for args in download_args
        }
        
        # Process completed downloads
        completed = 0
        for future in as_completed(future_to_instance):
            instance_name = future_to_instance[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                # Progress reporting
                if result['skipped']:
                    status = "⏭️ SKIP"
                elif result['success']:
                    status = "✅ OK"
                    if result['retry_count'] > 0:
                        status += f" (retry {result['retry_count']})"
                else:
                    status = "❌ FAIL"
                
                error_msg = f" - {result['error'][:40]}..." if result.get('error') else ""
                thread_safe_print(
                    f"[{completed:3d}/{len(instance_names)}] {status:15} {instance_name:15} "
                    f"{result.get('num_variables', 0):5d} vars, {result.get('num_constraints', 0):4d} cons"
                    f"{error_msg}"
                )
                
            except Exception as e:
                thread_safe_print(f"[{completed:3d}/{len(instance_names)}] ❌ ERROR      {instance_name:15} - {str(e)}")
                results.append({
                    'name': instance_name,
                    'success': False,
                    'error': str(e),
                    'skipped': False
                })
    
    total_time = time.time() - start_time
    
    # Compile statistics
    successful = [r for r in results if r['success'] and not r['skipped']]
    skipped = [r for r in results if r['skipped']]
    failed = [r for r in results if not r['success']]
    
    # Summary
    print("\n" + "=" * 70)
    print("PARALLEL DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total requested: {len(instance_names)}")
    print(f"Successfully downloaded: {len(successful)}")
    print(f"Skipped (already exist): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per download: {total_time/max(len(successful), 1):.1f}s")
    
    if successful:
        avg_vars = sum(r['num_variables'] for r in successful) / len(successful)
        avg_cons = sum(r['num_constraints'] for r in successful) / len(successful)
        print(f"Average size: {avg_vars:.0f} variables, {avg_cons:.0f} constraints")
    
    if failed:
        print(f"\nFAILED INSTANCES ({len(failed)}):")
        for result in failed:
            print(f"  {result['name']:20} - {result.get('error', 'Unknown error')}")
    
    return {
        'total': len(instance_names),
        'successful': len(successful),
        'skipped': len(skipped), 
        'failed': len(failed),
        'output_folder': base_folder,
        'total_time': total_time,
        'results': results
    }


def test_problematic_instances():
    """Test a comprehensive set to identify problematic instances."""
    test_instances = [
        # Basic scp instances
        'scp41.txt', 'scp42.txt', 'scp43.txt', 'scp44.txt', 'scp45.txt',
        'scp46.txt', 'scp47.txt', 'scp48.txt', 'scp49.txt', 'scp410.txt',
        'scp51.txt', 'scp52.txt', 'scp53.txt', 'scp54.txt', 'scp55.txt',
        'scp56.txt', 'scp57.txt', 'scp58.txt', 'scp59.txt', 'scp510.txt',
        'scp61.txt', 'scp62.txt', 'scp63.txt', 'scp64.txt', 'scp65.txt',
        
        # Lettered variants
        'scpa1.txt', 'scpa2.txt', 'scpa3.txt', 'scpa4.txt', 'scpa5.txt',
        'scpb1.txt', 'scpb2.txt', 'scpb3.txt', 'scpb4.txt', 'scpb5.txt',
        'scpc1.txt', 'scpc2.txt', 'scpc3.txt', 'scpc4.txt', 'scpc5.txt',
        'scpd1.txt', 'scpd2.txt', 'scpd3.txt', 'scpd4.txt', 'scpd5.txt',
        'scpe1.txt', 'scpe2.txt', 'scpe3.txt', 'scpe4.txt', 'scpe5.txt',
        
        # Special variants
        'scpclr10.txt', 'scpclr11.txt', 'scpclr12.txt', 'scpclr13.txt',
        'scpcyc06.txt', 'scpcyc07.txt', 'scpcyc08.txt', 'scpcyc09.txt',
        'scpcyc10.txt', 'scpcyc11.txt',
        
        # NR variants
        'scpnre1.txt', 'scpnre2.txt', 'scpnre3.txt', 'scpnre4.txt', 'scpnre5.txt',
        'scpnrf1.txt', 'scpnrf2.txt', 'scpnrf3.txt', 'scpnrf4.txt', 'scpnrf5.txt',
        'scpnrg1.txt', 'scpnrg2.txt', 'scpnrg3.txt', 'scpnrg4.txt', 'scpnrg5.txt',
        'scpnrh1.txt', 'scpnrh2.txt', 'scpnrh3.txt', 'scpnrh4.txt', 'scpnrh5.txt',
        
        # Rail instances
        'rail507.txt', 'rail516.txt', 'rail582.txt'
    ]
    
    return test_instances


def main():
    """Main entry point with enhanced command line interface."""
    parser = argparse.ArgumentParser(
        description="Download ORLIB set cover benchmark instances - Enhanced Parallel Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all instances with 8 parallel workers
  python benchmark_downloader.py --workers 8
  
  # Test comprehensive set to identify problematic instances
  python benchmark_downloader.py --test_comprehensive
  
  # Download specific instances
  python benchmark_downloader.py --instances scp41.txt rail507.txt
  
  # Force re-download with custom timeout
  python benchmark_downloader.py --force --timeout 60
        """
    )
    
    parser.add_argument("--output_dir", type=str, 
                       default='/Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization/data/setcover/benchmarks',
                       help="Directory to save downloaded instances")
    parser.add_argument("--instances", type=str, nargs="+", default=None,
                       help="Specific instance names to download (default: all)")
    parser.add_argument("--force", action="store_true",
                       help="Re-download existing files")
    parser.add_argument("--workers", type=int, default=6,
                       help="Number of parallel download workers (default: 6)")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout in seconds for each download (default: 30)")
    parser.add_argument("--max_retries", type=int, default=2,
                       help="Maximum retries for failed downloads (default: 2)")
    parser.add_argument("--test_comprehensive", action="store_true",
                       help="Test comprehensive set to identify problematic instances")
    
    args = parser.parse_args()
    
    # Determine which instances to download
    if args.test_comprehensive:
        instance_names = test_problematic_instances()
        print(f"Testing comprehensive set of {len(instance_names)} instances...")
    elif args.instances:
        instance_names = args.instances
        print(f"Downloading {len(instance_names)} specified instances...")
    else:
        instance_names = get_all_instance_names()
        print(f"Downloading all {len(instance_names)} available instances...")
    
    # Download instances
    results = download_instances_parallel(
        instance_names=instance_names,
        base_folder=args.output_dir,
        skip_existing=not args.force,
        max_workers=args.workers,
        timeout=args.timeout,
        max_retries=args.max_retries
    )
    
    # Final status
    if results['failed'] == 0:
        print("\n🎉 All downloads completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {results['failed']} downloads failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())

 