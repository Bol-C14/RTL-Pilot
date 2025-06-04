`timescale 1ns / 1ps

// Advanced Testbench for Synchronous FIFO
// RTL-Pilot Example demonstrating comprehensive verification

module tb_sync_fifo;

    // Parameters
    parameter DATA_WIDTH = 8;
    parameter FIFO_DEPTH = 16;
    parameter ADDR_WIDTH = $clog2(FIFO_DEPTH);
    parameter CLK_PERIOD = 10; // 100MHz
    
    // Testbench signals
    reg                  clk;
    reg                  rst_n;
    reg                  wr_en;
    reg [DATA_WIDTH-1:0] wr_data;
    wire                 full;
    wire                 almost_full;
    reg                  rd_en;
    wire [DATA_WIDTH-1:0] rd_data;
    wire                 empty;
    wire                 almost_empty;
    wire [ADDR_WIDTH:0]  count;
    
    // Test control signals
    reg [DATA_WIDTH-1:0] test_data_queue [$];
    reg [DATA_WIDTH-1:0] expected_data;
    integer test_num = 0;
    integer error_count = 0;
    integer transaction_count = 0;
    
    // Coverage tracking
    reg [DATA_WIDTH-1:0] written_data [$];
    reg [DATA_WIDTH-1:0] read_data [$];

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT instantiation
    sync_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .wr_data(wr_data),
        .full(full),
        .almost_full(almost_full),
        .rd_en(rd_en),
        .rd_data(rd_data),
        .empty(empty),
        .almost_empty(almost_empty),
        .count(count)
    );

    // Reset task
    task reset_dut;
        begin
            rst_n = 0;
            wr_en = 0;
            rd_en = 0;
            wr_data = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask

    // Write data task
    task write_data;
        input [DATA_WIDTH-1:0] data;
        begin
            if (!full) begin
                @(posedge clk);
                wr_en = 1;
                wr_data = data;
                test_data_queue.push_back(data);
                written_data.push_back(data);
                @(posedge clk);
                wr_en = 0;
                transaction_count++;
                $display("Time %0t: Wrote data 0x%02x, Count: %0d", $time, data, count);
            end else begin
                $display("Time %0t: Cannot write, FIFO full", $time);
            end
        end
    endtask

    // Read data task
    task read_data;
        output [DATA_WIDTH-1:0] data;
        begin
            if (!empty) begin
                expected_data = test_data_queue.pop_front();
                @(posedge clk);
                rd_en = 1;
                @(posedge clk);
                rd_en = 0;
                data = rd_data;
                read_data.push_back(data);
                transaction_count++;
                
                if (data === expected_data) begin
                    $display("Time %0t: Read data 0x%02x (Expected: 0x%02x) - PASS", 
                            $time, data, expected_data);
                end else begin
                    $display("Time %0t: Read data 0x%02x (Expected: 0x%02x) - FAIL", 
                            $time, data, expected_data);
                    error_count++;
                end
            end else begin
                $display("Time %0t: Cannot read, FIFO empty", $time);
                data = 8'hxx;
            end
        end
    endtask

    // Test: Basic functionality
    task test_basic_functionality;
        reg [DATA_WIDTH-1:0] read_val;
        begin
            test_num++;
            $display("\n=== Test %0d: Basic Functionality ===", test_num);
            
            reset_dut();
            
            // Write some data
            write_data(8'h12);
            write_data(8'h34);
            write_data(8'h56);
            
            // Read back data
            read_data(read_val);
            read_data(read_val);
            read_data(read_val);
        end
    endtask

    // Test: Full/Empty conditions
    task test_full_empty;
        reg [DATA_WIDTH-1:0] read_val;
        integer i;
        begin
            test_num++;
            $display("\n=== Test %0d: Full/Empty Conditions ===", test_num);
            
            reset_dut();
            
            // Fill the FIFO completely
            $display("Filling FIFO...");
            for (i = 0; i < FIFO_DEPTH; i++) begin
                write_data(i);
            end
            
            // Verify full condition
            if (!full) begin
                $display("ERROR: FIFO should be full but full flag is not set");
                error_count++;
            end
            
            // Try to write when full (should not work)
            @(posedge clk);
            wr_en = 1;
            wr_data = 8'hFF;
            @(posedge clk);
            wr_en = 0;
            
            // Empty the FIFO completely
            $display("Emptying FIFO...");
            for (i = 0; i < FIFO_DEPTH; i++) begin
                read_data(read_val);
            end
            
            // Verify empty condition
            if (!empty) begin
                $display("ERROR: FIFO should be empty but empty flag is not set");
                error_count++;
            end
            
            // Try to read when empty (should not work)
            @(posedge clk);
            rd_en = 1;
            @(posedge clk);
            rd_en = 0;
        end
    endtask

    // Test: Almost full/empty conditions
    task test_almost_flags;
        reg [DATA_WIDTH-1:0] read_val;
        integer i;
        begin
            test_num++;
            $display("\n=== Test %0d: Almost Full/Empty Flags ===", test_num);
            
            reset_dut();
            
            // Fill to almost full
            for (i = 0; i < FIFO_DEPTH-1; i++) begin
                write_data(i);
            end
            
            if (!almost_full) begin
                $display("ERROR: Almost full flag should be set");
                error_count++;
            end
            
            // Add one more to make it full
            write_data(8'hFF);
            
            // Read one item
            read_data(read_val);
            
            if (!almost_full) begin
                $display("ERROR: Almost full flag should still be set");
                error_count++;
            end
            
            // Empty to almost empty
            for (i = 0; i < FIFO_DEPTH-2; i++) begin
                read_data(read_val);
            end
            
            if (!almost_empty) begin
                $display("ERROR: Almost empty flag should be set");
                error_count++;
            end
        end
    endtask

    // Test: Concurrent read/write operations
    task test_concurrent_operations;
        reg [DATA_WIDTH-1:0] read_val;
        integer i;
        begin
            test_num++;
            $display("\n=== Test %0d: Concurrent Read/Write ===", test_num);
            
            reset_dut();
            
            // Fill FIFO halfway
            for (i = 0; i < FIFO_DEPTH/2; i++) begin
                write_data(i);
            end
            
            // Concurrent operations
            for (i = 0; i < 20; i++) begin
                fork
                    begin
                        if (!full && ($random % 2)) begin
                            write_data($random & 8'hFF);
                        end
                    end
                    begin
                        if (!empty && ($random % 2)) begin
                            read_data(read_val);
                        end
                    end
                join
                @(posedge clk);
            end
        end
    endtask

    // Test: Random stress test
    task test_random_stress;
        reg [DATA_WIDTH-1:0] read_val;
        integer i, wr_prob, rd_prob;
        begin
            test_num++;
            $display("\n=== Test %0d: Random Stress Test ===", test_num);
            
            reset_dut();
            
            for (i = 0; i < 1000; i++) begin
                wr_prob = $random % 100;
                rd_prob = $random % 100;
                
                // Random write
                if (wr_prob < 60 && !full) begin
                    write_data($random & 8'hFF);
                end
                
                // Random read
                if (rd_prob < 40 && !empty) begin
                    read_data(read_val);
                end
                
                @(posedge clk);
            end
        end
    endtask

    // Main test sequence
    initial begin
        $display("Starting FIFO Testbench");
        $display("======================");
        $display("DATA_WIDTH: %0d, FIFO_DEPTH: %0d", DATA_WIDTH, FIFO_DEPTH);
        
        // Run all tests
        test_basic_functionality();
        test_full_empty();
        test_almost_flags();
        test_concurrent_operations();
        test_random_stress();
        
        // Empty remaining queue
        while (test_data_queue.size() > 0) begin
            automatic reg [DATA_WIDTH-1:0] dummy = test_data_queue.pop_front();
        end
        
        // Final report
        #100;
        $display("\n======================");
        $display("Test Summary:");
        $display("Tests run: %0d", test_num);
        $display("Transactions: %0d", transaction_count);
        $display("Errors: %0d", error_count);
        $display("Data written: %0d items", written_data.size());
        $display("Data read: %0d items", read_data.size());
        
        if (error_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("TESTS FAILED!");
        end
        
        $display("Simulation completed at time %0t", $time);
        $finish;
    end

    // Timeout watchdog
    initial begin
        #1000000; // 1ms timeout
        $display("ERROR: Testbench timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("fifo_waves.vcd");
        $dumpvars(0, tb_sync_fifo);
    end

    // Monitor FIFO status
    always @(posedge clk) begin
        if (rst_n) begin
            if (count > FIFO_DEPTH) begin
                $display("ERROR: FIFO count overflow at time %0t", $time);
                error_count++;
            end
        end
    end

endmodule
