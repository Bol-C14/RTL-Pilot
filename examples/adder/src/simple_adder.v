// Simple 4-bit Adder Design
// RTL-Pilot Example

module simple_adder (
    input  wire [3:0] a,
    input  wire [3:0] b,
    input  wire       cin,
    output wire [3:0] sum,
    output wire       cout
);

    // Internal carry signals
    wire [4:0] carry;
    
    // Connect input carry
    assign carry[0] = cin;
    
    // Generate full adder for each bit
    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : adder_gen
            full_adder fa_inst (
                .a(a[i]),
                .b(b[i]),
                .cin(carry[i]),
                .sum(sum[i]),
                .cout(carry[i+1])
            );
        end
    endgenerate
    
    // Output carry
    assign cout = carry[4];

endmodule

// Full Adder Module
module full_adder (
    input  wire a,
    input  wire b,
    input  wire cin,
    output wire sum,
    output wire cout
);

    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | (a & cin) | (b & cin);

endmodule
