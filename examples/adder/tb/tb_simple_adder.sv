`timescale 1ns / 1ps

// Testbench for Simple Adder
// RTL-Pilot Example

module tb_simple_adder;

    // Testbench signals
    reg [3:0] a;
    reg [3:0] b;
    reg       cin;
    wire [3:0] sum;
    wire       cout;
    
    // Test result tracking
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;
    
    // Expected results
    reg [4:0] expected_result;
    wire [4:0] actual_result;
    assign actual_result = {cout, sum};

    // Instantiate the Device Under Test (DUT)
    simple_adder dut (
        .a(a),
        .b(b),
        .cin(cin),
        .sum(sum),
        .cout(cout)
    );

    // Test procedure
    task run_test;
        input [3:0] test_a;
        input [3:0] test_b;
        input test_cin;
        input [4:0] expected;
        begin
            a = test_a;
            b = test_b;
            cin = test_cin;
            expected_result = expected;
            
            #10; // Wait for propagation
            
            test_count = test_count + 1;
            
            if (actual_result === expected_result) begin
                $display("PASS Test %0d: %0d + %0d + %0d = %0d (Expected: %0d)", 
                        test_count, a, b, cin, actual_result, expected_result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL Test %0d: %0d + %0d + %0d = %0d (Expected: %0d)", 
                        test_count, a, b, cin, actual_result, expected_result);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Main test sequence
    initial begin
        $display("Starting Simple Adder Testbench");
        $display("================================");
        
        // Initialize inputs
        a = 0;
        b = 0;
        cin = 0;
        
        #5; // Initial delay
        
        // Test Case 1: Basic addition without carry
        run_test(4'b0001, 4'b0010, 1'b0, 5'b00011); // 1 + 2 + 0 = 3
        
        // Test Case 2: Addition with input carry
        run_test(4'b0001, 4'b0010, 1'b1, 5'b00100); // 1 + 2 + 1 = 4
        
        // Test Case 3: Addition with output carry
        run_test(4'b1111, 4'b0001, 1'b0, 5'b10000); // 15 + 1 + 0 = 16 (overflow)
        
        // Test Case 4: Maximum values
        run_test(4'b1111, 4'b1111, 1'b1, 5'b11111); // 15 + 15 + 1 = 31
        
        // Test Case 5: Zero addition
        run_test(4'b0000, 4'b0000, 1'b0, 5'b00000); // 0 + 0 + 0 = 0
        
        // Test Case 6: Carry propagation test
        run_test(4'b0111, 4'b0001, 1'b0, 5'b01000); // 7 + 1 + 0 = 8
        
        // Test Case 7: Another carry test
        run_test(4'b1000, 4'b1000, 1'b0, 5'b10000); // 8 + 8 + 0 = 16
        
        // Test Case 8: Random test
        run_test(4'b0101, 4'b1010, 1'b1, 5'b10000); // 5 + 10 + 1 = 16
        
        // Comprehensive test - all combinations (optional, long test)
        $display("\nRunning comprehensive test...");
        automatic integer total_tests = 0;
        automatic integer total_pass = 0;
        
        for (int test_a = 0; test_a < 16; test_a++) begin
            for (int test_b = 0; test_b < 16; test_b++) begin
                for (int test_cin = 0; test_cin < 2; test_cin++) begin
                    automatic int expected = test_a + test_b + test_cin;
                    run_test(test_a, test_b, test_cin, expected);
                    total_tests++;
                    if (actual_result === expected) total_pass++;
                end
            end
        end
        
        // Final summary
        #10;
        $display("\n================================");
        $display("Testbench Summary:");
        $display("Total Tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        $display("Pass Rate: %0.1f%%", (pass_count * 100.0) / test_count);
        
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        
        $display("Testbench completed at time %0t", $time);
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000; // 100us timeout
        $display("ERROR: Testbench timeout!");
        $finish;
    end
    
    // Waveform dump for analysis
    initial begin
        $dumpfile("adder_waves.vcd");
        $dumpvars(0, tb_simple_adder);
    end

endmodule
