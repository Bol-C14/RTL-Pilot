// Synchronous FIFO Design
// RTL-Pilot Advanced Example

module sync_fifo #(
    parameter DATA_WIDTH = 8,
    parameter FIFO_DEPTH = 16,
    parameter ADDR_WIDTH = $clog2(FIFO_DEPTH)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    
    // Write interface
    input  wire                  wr_en,
    input  wire [DATA_WIDTH-1:0] wr_data,
    output wire                  full,
    output wire                  almost_full,
    
    // Read interface
    input  wire                  rd_en,
    output wire [DATA_WIDTH-1:0] rd_data,
    output wire                  empty,
    output wire                  almost_empty,
    
    // Status signals
    output wire [ADDR_WIDTH:0]   count
);

    // Internal memory array
    reg [DATA_WIDTH-1:0] memory [0:FIFO_DEPTH-1];
    
    // Pointers
    reg [ADDR_WIDTH:0] wr_ptr;
    reg [ADDR_WIDTH:0] rd_ptr;
    
    // Internal signals
    wire wr_enable, rd_enable;
    wire [ADDR_WIDTH-1:0] wr_addr, rd_addr;
    
    // Enable signals
    assign wr_enable = wr_en && !full;
    assign rd_enable = rd_en && !empty;
    
    // Address extraction
    assign wr_addr = wr_ptr[ADDR_WIDTH-1:0];
    assign rd_addr = rd_ptr[ADDR_WIDTH-1:0];
    
    // FIFO count
    assign count = wr_ptr - rd_ptr;
    
    // Status flags
    assign empty = (count == 0);
    assign full = (count == FIFO_DEPTH);
    assign almost_empty = (count == 1);
    assign almost_full = (count == (FIFO_DEPTH - 1));
    
    // Write operation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else if (wr_enable) begin
            memory[wr_addr] <= wr_data;
            wr_ptr <= wr_ptr + 1;
        end
    end
    
    // Read operation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= 0;
        end else if (rd_enable) begin
            rd_ptr <= rd_ptr + 1;
        end
    end
    
    // Read data output (combinational)
    assign rd_data = memory[rd_addr];
    
    // Assertions for design verification
    `ifdef SIMULATION
        // Check for overflow
        always @(posedge clk) begin
            if (rst_n && wr_en && full) begin
                $error("FIFO overflow detected!");
            end
        end
        
        // Check for underflow
        always @(posedge clk) begin
            if (rst_n && rd_en && empty) begin
                $error("FIFO underflow detected!");
            end
        end
        
        // Check pointer consistency
        always @(posedge clk) begin
            if (rst_n && (count > FIFO_DEPTH)) begin
                $error("FIFO count overflow!");
            end
        end
    `endif

endmodule
