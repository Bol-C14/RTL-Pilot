# RTL-Pilot Vivado Automation Script
# This script provides standardized procedures for RTL verification workflows
# 
# Usage: vivado -mode batch -source vivado_run.tcl -tclargs <command> [args...]
# 
# Available commands:
#   create_project <name> <part> <rtl_files> <tb_files>
#   run_simulation <project> [time]
#   analyze_results <project>
#   generate_reports <project>

# Global variables
set script_name "RTL-Pilot Vivado Runner"
set script_version "1.0"

# Parse command line arguments
if {$argc < 1} {
    puts "ERROR: No command specified"
    puts "Usage: vivado -mode batch -source vivado_run.tcl -tclargs <command> \[args...\]"
    exit 1
}

set command [lindex $argv 0]
set args [lrange $argv 1 end]

puts "INFO: $script_name v$script_version"
puts "INFO: Executing command: $command"

# Helper procedures
proc log_info {msg} {
    puts "INFO: $msg"
}

proc log_warning {msg} {
    puts "WARNING: $msg"
}

proc log_error {msg} {
    puts "ERROR: $msg"
}

proc check_file_exists {filepath} {
    if {![file exists $filepath]} {
        log_error "File not found: $filepath"
        return 0
    }
    return 1
}

# Create project procedure
proc create_project_proc {project_name target_part rtl_files tb_files} {
    log_info "Creating project: $project_name"
    log_info "Target part: $target_part"
    
    # Create the project
    if {[catch {create_project $project_name . -part $target_part -force} result]} {
        log_error "Failed to create project: $result"
        return 0
    }
    
    # Add RTL files
    foreach rtl_file $rtl_files {
        if {[check_file_exists $rtl_file]} {
            log_info "Adding RTL file: $rtl_file"
            add_files $rtl_file
        } else {
            log_warning "Skipping missing RTL file: $rtl_file"
        }
    }
    
    # Add testbench files
    foreach tb_file $tb_files {
        if {[check_file_exists $tb_file]} {
            log_info "Adding testbench file: $tb_file"
            add_files -fileset sim_1 $tb_file
        } else {
            log_warning "Skipping missing testbench file: $tb_file"
        }
    }
    
    # Set simulation top module (use first testbench file name without extension)
    if {[llength $tb_files] > 0} {
        set first_tb [lindex $tb_files 0]
        set tb_top [file rootname [file tail $first_tb]]
        log_info "Setting simulation top module: $tb_top"
        set_property top $tb_top [get_filesets sim_1]
        set_property top_lib xil_defaultlib [get_filesets sim_1]
    }
    
    # Update compile order
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    
    log_info "Project created successfully: $project_name"
    return 1
}

# Run simulation procedure
proc run_simulation_proc {project_name {sim_time "1000ns"}} {
    log_info "Running simulation for project: $project_name"
    log_info "Simulation time: $sim_time"
    
    # Check if project exists
    set project_file "${project_name}.xpr"
    if {![check_file_exists $project_file]} {
        return 0
    }
    
    # Open project
    if {[catch {open_project $project_file} result]} {
        log_error "Failed to open project: $result"
        return 0
    }
    
    # Launch simulation
    if {[catch {launch_simulation} result]} {
        log_error "Failed to launch simulation: $result"
        return 0
    }
    
    # Run simulation for specified time
    if {[catch {run $sim_time} result]} {
        log_error "Simulation failed: $result"
        return 0
    }
    
    log_info "Simulation completed successfully"
    
    # Save waveform if GUI is available
    if {[catch {save_wave_config "${project_name}_wave.wcfg"} result]} {
        log_warning "Could not save waveform: $result"
    }
    
    return 1
}

# Analyze simulation results procedure
proc analyze_results_proc {project_name} {
    log_info "Analyzing results for project: $project_name"
    
    # Check for common result files
    set result_files [list \
        "${project_name}.sim/sim_1/behav/xsim/simulate.log" \
        "${project_name}.sim/sim_1/behav/xsim/compile.log" \
        "${project_name}.sim/sim_1/behav/xsim/elaborate.log" \
    ]
    
    set analysis_results [dict create]
    
    foreach result_file $result_files {
        if {[file exists $result_file]} {
            log_info "Found result file: $result_file"
            
            # Read and analyze the file
            set fp [open $result_file r]
            set content [read $fp]
            close $fp
            
            # Count errors and warnings
            set error_count [regexp -all -nocase {error} $content]
            set warning_count [regexp -all -nocase {warning} $content]
            
            dict set analysis_results $result_file [dict create \
                "errors" $error_count \
                "warnings" $warning_count \
                "size" [string length $content] \
            ]
            
            log_info "File $result_file: $error_count errors, $warning_count warnings"
        }
    }
    
    # Generate analysis summary
    set summary_file "${project_name}_analysis.txt"
    set fp [open $summary_file w]
    puts $fp "RTL-Pilot Simulation Analysis Report"
    puts $fp "Project: $project_name"
    puts $fp "Timestamp: [clock format [clock seconds]]"
    puts $fp "="
    
    dict for {filename stats} $analysis_results {
        puts $fp "\nFile: $filename"
        puts $fp "  Errors: [dict get $stats errors]"
        puts $fp "  Warnings: [dict get $stats warnings]"
        puts $fp "  Size: [dict get $stats size] bytes"
    }
    
    close $fp
    log_info "Analysis report saved: $summary_file"
    
    return 1
}

# Generate reports procedure
proc generate_reports_proc {project_name} {
    log_info "Generating reports for project: $project_name"
    
    set project_file "${project_name}.xpr"
    if {![check_file_exists $project_file]} {
        return 0
    }
    
    # Open project if not already open
    if {[catch {open_project $project_file} result]} {
        log_warning "Project may already be open: $result"
    }
    
    # Create reports directory
    set reports_dir "${project_name}_reports"
    file mkdir $reports_dir
    
    # Generate utilization report (if implemented)
    if {[catch {
        if {[get_runs -quiet impl_1] != ""} {
            report_utilization -file "$reports_dir/utilization.rpt"
            log_info "Utilization report generated"
        }
    } result]} {
        log_warning "Could not generate utilization report: $result"
    }
    
    # Generate timing report (if implemented)  
    if {[catch {
        if {[get_runs -quiet impl_1] != ""} {
            report_timing_summary -file "$reports_dir/timing.rpt"
            log_info "Timing report generated"
        }
    } result]} {
        log_warning "Could not generate timing report: $result"
    }
    
    # Generate DRC report
    if {[catch {
        report_drc -file "$reports_dir/drc.rpt"
        log_info "DRC report generated"
    } result]} {
        log_warning "Could not generate DRC report: $result"
    }
    
    log_info "Reports saved in directory: $reports_dir"
    return 1
}

# Command dispatch
switch $command {
    "create_project" {
        if {[llength $args] < 4} {
            log_error "create_project requires: <name> <part> <rtl_files> <tb_files>"
            exit 1
        }
        
        set project_name [lindex $args 0]
        set target_part [lindex $args 1]
        set rtl_files [split [lindex $args 2] ","]
        set tb_files [split [lindex $args 3] ","]
        
        if {![create_project_proc $project_name $target_part $rtl_files $tb_files]} {
            exit 1
        }
    }
    
    "run_simulation" {
        if {[llength $args] < 1} {
            log_error "run_simulation requires: <project> \[time\]"
            exit 1
        }
        
        set project_name [lindex $args 0]
        set sim_time "1000ns"
        if {[llength $args] > 1} {
            set sim_time [lindex $args 1]
        }
        
        if {![run_simulation_proc $project_name $sim_time]} {
            exit 1
        }
    }
    
    "analyze_results" {
        if {[llength $args] < 1} {
            log_error "analyze_results requires: <project>"
            exit 1
        }
        
        set project_name [lindex $args 0]
        
        if {![analyze_results_proc $project_name]} {
            exit 1
        }
    }
    
    "generate_reports" {
        if {[llength $args] < 1} {
            log_error "generate_reports requires: <project>"
            exit 1
        }
        
        set project_name [lindex $args 0]
        
        if {![generate_reports_proc $project_name]} {
            exit 1
        }
    }
    
    "version" {
        puts "Vivado version: [version -short]"
        puts "$script_name v$script_version"
    }
    
    default {
        log_error "Unknown command: $command"
        log_error "Available commands: create_project, run_simulation, analyze_results, generate_reports, version"
        exit 1
    }
}

log_info "Command completed successfully"
exit 0
