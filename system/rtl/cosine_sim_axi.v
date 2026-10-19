// =============================================================================
// cosine_sim_axi.v
// AXI4-Lite slave wrapper around cosine_similarity.v
// -----------------------------------------------------------------------------
// Register Map (all 32-bit, byte-addressable, 4-byte aligned):
//
//  Offset  Name          Access  Description
//  0x00    CTRL          W       [0]=START pulse (write 1 to begin a transfer)
//  0x04    STATUS        R       [0]=BUSY, [1]=RESULT_VALID, [2]=MATCH
//  0x08    THRESHOLD     RW      Q2.30 threshold (default 0x66666666 = 0.80)
//  0x0C    VEC_SEL       W       [0]=0→write A, 1→write B; auto-clears after N writes
//  0x10    VEC_DATA      W       Write one Q1.15 sample (lower 16 bits) per write
//                                First write after VEC_SEL loads element[0],
//                                subsequent writes fill element[1..127] in order.
//  0x14    LOAD_COUNT    R       How many elements loaded into current vector (0-128)
// =============================================================================

`timescale 1ns/1ps

module cosine_sim_axi #(
    parameter N    = 128,
    parameter DW   = 16,
    parameter ACCW = 48,
    parameter [31:0] DEFAULT_THRESHOLD = 32'h33333333,
    // AXI
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 5    // 32 bytes of reg space
)(
    // AXI4-Lite slave
    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    S_AXI_AWADDR,
    input  wire [2:0]                        S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output reg                               S_AXI_AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]    S_AXI_WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]  S_AXI_WSTRB,
    input  wire                              S_AXI_WVALID,
    output reg                               S_AXI_WREADY,
    output reg  [1:0]                        S_AXI_BRESP,
    output reg                               S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    S_AXI_ARADDR,
    input  wire [2:0]                        S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output reg                               S_AXI_ARREADY,
    output reg  [C_S_AXI_DATA_WIDTH-1:0]    S_AXI_RDATA,
    output reg  [1:0]                        S_AXI_RRESP,
    output reg                               S_AXI_RVALID,
    input  wire                              S_AXI_RREADY,

    // Auth output - stable '1' when cosine >= threshold, '0' otherwise.
    // Holds its value until the next START pulse clears it.
    // Wire this directly into your combinational auth module.
    output reg                               auth_valid
);

// ---------------------------------------------------------------------------
// Internal registers
// ---------------------------------------------------------------------------
reg [31:0] reg_threshold;
reg        reg_vec_sel;       // 0=A, 1=B
reg [31:0] reg_status;        // [0]=busy [1]=result_valid [2]=match

// Vector RAMs  (128 × 16-bit each)
reg signed [DW-1:0] ram_a [0:N-1];
reg signed [DW-1:0] ram_b [0:N-1];

reg [$clog2(N):0] load_cnt_a, load_cnt_b;  // how many elements loaded
reg               load_done_a, load_done_b;

// Cosine core signals
reg                   cs_valid_in;
reg signed [DW-1:0]   cs_a_in, cs_b_in;
wire                  cs_match, cs_result_valid;
reg [$clog2(N)-1:0]   feed_idx;
reg                   feeding;
reg                   busy;

// ---------------------------------------------------------------------------
// Instantiate core
// ---------------------------------------------------------------------------
cosine_similarity #(
    .N(N), .DW(DW), .ACCW(ACCW), .THRESHOLD(DEFAULT_THRESHOLD)
) core (
    .clk          (S_AXI_ACLK),
    .rst_n        (S_AXI_ARESETN),
    .a_in         (cs_a_in),
    .b_in         (cs_b_in),
    .valid_in     (cs_valid_in),
    .threshold    (reg_threshold),
    .match        (cs_match),
    .result_valid (cs_result_valid)
);

// ---------------------------------------------------------------------------
// AXI write channel
// ---------------------------------------------------------------------------
reg [C_S_AXI_ADDR_WIDTH-1:0] aw_addr_lat;
reg                            aw_valid_lat;

// Latch write address
always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        S_AXI_AWREADY  <= 0;
        aw_addr_lat    <= 0;
        aw_valid_lat   <= 0;
    end else begin
        if (!S_AXI_AWREADY && S_AXI_AWVALID && S_AXI_WVALID) begin
            S_AXI_AWREADY <= 1;
            aw_addr_lat   <= S_AXI_AWADDR;
            aw_valid_lat  <= 1;
        end else begin
            S_AXI_AWREADY <= 0;
        end
    end
end

// Latch write data
always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        S_AXI_WREADY <= 0;
    end else begin
        if (!S_AXI_WREADY && S_AXI_WVALID && S_AXI_AWVALID)
            S_AXI_WREADY <= 1;
        else
            S_AXI_WREADY <= 0;
    end
end

wire write_en = S_AXI_WREADY && S_AXI_WVALID && S_AXI_AWREADY && S_AXI_AWVALID;

// Write response
always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        S_AXI_BVALID <= 0;
        S_AXI_BRESP  <= 2'b00;
    end else begin
        if (write_en && !S_AXI_BVALID) begin
            S_AXI_BVALID <= 1;
            S_AXI_BRESP  <= 2'b00;  // OKAY
        end else if (S_AXI_BREADY && S_AXI_BVALID) begin
            S_AXI_BVALID <= 0;
        end
    end
end

// ---------------------------------------------------------------------------
// Register writes
// ---------------------------------------------------------------------------
integer ii;

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        reg_threshold <= DEFAULT_THRESHOLD;
        reg_vec_sel   <= 0;
        load_cnt_a    <= 0;
        load_cnt_b    <= 0;
        load_done_a   <= 0;
        load_done_b   <= 0;
        // zero rams
        for (ii = 0; ii < N; ii = ii+1) begin
            ram_a[ii] <= 0;
            ram_b[ii] <= 0;
        end
    end else if (write_en) begin
        case (aw_addr_lat[4:2])   // word address bits [4:2]
            3'h0: begin  // CTRL - START is handled in the feeder FSM
            end
            3'h2: begin  // THRESHOLD
                reg_threshold <= S_AXI_WDATA;
            end
            3'h3: begin  // VEC_SEL
                reg_vec_sel <= S_AXI_WDATA[0];
                if (S_AXI_WDATA[0] == 0) begin
                    load_cnt_a  <= 0;
                    load_done_a <= 0;
                end else begin
                    load_cnt_b  <= 0;
                    load_done_b <= 0;
                end
            end
            3'h4: begin  // VEC_DATA
                if (reg_vec_sel == 0 && !load_done_a) begin
                    ram_a[load_cnt_a] <= S_AXI_WDATA[DW-1:0];
                    if (load_cnt_a == N-1) begin
                        load_done_a <= 1;
                        load_cnt_a  <= 0;
                    end else begin
                        load_cnt_a <= load_cnt_a + 1;
                    end
                end else if (reg_vec_sel == 1 && !load_done_b) begin
                    ram_b[load_cnt_b] <= S_AXI_WDATA[DW-1:0];
                    if (load_cnt_b == N-1) begin
                        load_done_b <= 1;
                        load_cnt_b  <= 0;
                    end else begin
                        load_cnt_b <= load_cnt_b + 1;
                    end
                end
            end
            default: ;
        endcase
    end
end

// ---------------------------------------------------------------------------
// Feeding FSM  - triggered by CTRL[0]=1 write
// ---------------------------------------------------------------------------
wire start_pulse = write_en && (aw_addr_lat[4:2] == 3'h0) && S_AXI_WDATA[0];

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        feeding    <= 0;
        busy       <= 0;
        feed_idx   <= 0;
        cs_valid_in <= 0;
        cs_a_in    <= 0;
        cs_b_in    <= 0;
        reg_status <= 0;
        auth_valid <= 0;
    end else begin
        cs_valid_in <= 0;

        // Capture result
        if (cs_result_valid) begin
            reg_status[1] <= 1;
            reg_status[2] <= cs_match;
            reg_status[0] <= 0;  // clear busy
            busy          <= 0;
            auth_valid    <= cs_match;  // latch: stays until next START
        end

        if (start_pulse && !busy) begin
            feeding      <= 1;
            busy         <= 1;
            feed_idx     <= 0;
            reg_status   <= 3'b001;  // busy, clear old result
            auth_valid   <= 0;       // clear while computing
        end

        if (feeding) begin
            cs_a_in     <= ram_a[feed_idx];
            cs_b_in     <= ram_b[feed_idx];
            cs_valid_in <= 1;
            if (feed_idx == N-1) begin
                feeding  <= 0;
                feed_idx <= 0;
            end else begin
                feed_idx <= feed_idx + 1;
            end
        end
    end
end

// ---------------------------------------------------------------------------
// AXI read channel
// ---------------------------------------------------------------------------
always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        S_AXI_ARREADY <= 0;
        S_AXI_RVALID  <= 0;
        S_AXI_RRESP   <= 0;
        S_AXI_RDATA   <= 0;
    end else begin
        if (!S_AXI_ARREADY && S_AXI_ARVALID) begin
            S_AXI_ARREADY <= 1;
            case (S_AXI_ARADDR[4:2])
                3'h1: S_AXI_RDATA <= reg_status;
                3'h2: S_AXI_RDATA <= reg_threshold;
                3'h5: S_AXI_RDATA <= (reg_vec_sel == 0)
                                     ? {22'b0, load_cnt_a}
                                     : {22'b0, load_cnt_b};
                default: S_AXI_RDATA <= 32'hDEADBEEF;
            endcase
            S_AXI_RRESP  <= 2'b00;
            S_AXI_RVALID <= 1;
        end else begin
            S_AXI_ARREADY <= 0;
            if (S_AXI_RVALID && S_AXI_RREADY)
                S_AXI_RVALID <= 0;
        end
    end
end

endmodule