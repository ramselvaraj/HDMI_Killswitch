// =============================================================================
// cosine_similarity.v
// -----------------------------------------------------------------------------
// Pipelined cosine-similarity engine for two N-element vectors.
//
// Data format : Q1.15 fixed-point, signed 16-bit
// Threshold   : Q2.30 fixed-point
//
// Computes:
//   dot(A,B)^2 >= threshold^2 * dot(A,A) * dot(B,B)
// and also requires dot(A,B) >= 0 for positive thresholds.
// =============================================================================

`timescale 1ns / 1ps

module cosine_similarity #(
    parameter N          = 128,
    parameter DW         = 16,
    parameter ACCW       = 48,
    parameter THRESHOLD  = 32'h33333333
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire signed [DW-1:0] a_in,
    input  wire signed [DW-1:0] b_in,
    input  wire                 valid_in,

    input  wire [31:0]          threshold,

    output reg                  match,
    output reg                  result_valid
);

localparam PROD_W = 2 * DW;

// ---------------------------------------------------------------------------
// 1. Input multiply stage
// ---------------------------------------------------------------------------
reg signed [PROD_W-1:0] p_ab;
reg signed [PROD_W-1:0] p_aa;
reg signed [PROD_W-1:0] p_bb;
reg                     p_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        p_ab    <= 0;
        p_aa    <= 0;
        p_bb    <= 0;
        p_valid <= 1'b0;
    end else begin
        p_ab    <= a_in * b_in;
        p_aa    <= a_in * a_in;
        p_bb    <= b_in * b_in;
        p_valid <= valid_in;
    end
end

// ---------------------------------------------------------------------------
// 2. Accumulation stage
// ---------------------------------------------------------------------------
reg signed [ACCW-1:0] acc_ab;
reg signed [ACCW-1:0] acc_aa;
reg signed [ACCW-1:0] acc_bb;

reg [$clog2(N):0] acc_cnt;
reg               acc_done;

wire signed [ACCW-1:0] p_ab_ext =
    {{(ACCW-PROD_W){p_ab[PROD_W-1]}}, p_ab};

wire signed [ACCW-1:0] p_aa_ext =
    {{(ACCW-PROD_W){p_aa[PROD_W-1]}}, p_aa};

wire signed [ACCW-1:0] p_bb_ext =
    {{(ACCW-PROD_W){p_bb[PROD_W-1]}}, p_bb};

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        acc_ab   <= 0;
        acc_aa   <= 0;
        acc_bb   <= 0;
        acc_cnt  <= 0;
        acc_done <= 1'b0;
    end else begin
        acc_done <= 1'b0;

        if (p_valid) begin
            if (acc_cnt == 0) begin
                acc_ab <= p_ab_ext;
                acc_aa <= p_aa_ext;
                acc_bb <= p_bb_ext;
            end else begin
                acc_ab <= acc_ab + p_ab_ext;
                acc_aa <= acc_aa + p_aa_ext;
                acc_bb <= acc_bb + p_bb_ext;
            end

            if (acc_cnt == N-1) begin
                acc_cnt  <= 0;
                acc_done <= 1'b1;
            end else begin
                acc_cnt <= acc_cnt + 1'b1;
            end
        end else begin
            acc_cnt <= 0;
        end
    end
end

// ---------------------------------------------------------------------------
// 3. Latch final accumulators
// ---------------------------------------------------------------------------
reg signed [ACCW-1:0] lat_ab;
reg        [ACCW-1:0] lat_aa;
reg        [ACCW-1:0] lat_bb;
reg                   lat_valid;
reg [31:0]            lat_threshold;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        lat_ab        <= 0;
        lat_aa        <= 0;
        lat_bb        <= 0;
        lat_valid     <= 1'b0;
        lat_threshold <= THRESHOLD;
    end else begin
        lat_valid <= acc_done;

        if (acc_done) begin
            lat_ab        <= acc_ab;
            lat_aa        <= acc_aa[ACCW-1] ? 0 : acc_aa;
            lat_bb        <= acc_bb[ACCW-1] ? 0 : acc_bb;
            lat_threshold <= threshold;
        end
    end
end

// ---------------------------------------------------------------------------
// 4. Compare pipeline stage 1
//
// Large multiplies are registered here.
// ---------------------------------------------------------------------------
reg [63:0] thr_sq_s1;
reg [95:0] dot_sq_full_s1;
reg [95:0] norm_sq_full_s1;
reg        dot_nonneg_s1;
reg        cmp_valid_s1;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        thr_sq_s1       <= 64'd0;
        dot_sq_full_s1  <= 96'd0;
        norm_sq_full_s1 <= 96'd0;
        dot_nonneg_s1   <= 1'b0;
        cmp_valid_s1    <= 1'b0;
    end else begin
        cmp_valid_s1 <= lat_valid;

        if (lat_valid) begin
            thr_sq_s1       <= lat_threshold * lat_threshold;
            dot_sq_full_s1  <= lat_ab * lat_ab;
            norm_sq_full_s1 <= lat_aa * lat_bb;
            dot_nonneg_s1   <= !lat_ab[ACCW-1];
        end
    end
end

// ---------------------------------------------------------------------------
// 5. Compare pipeline stage 2
//
// Preserve your original scaling:
//   dot_sq  = dot_sq_full[95:44]
//   norm_sq = norm_sq_full[95:44]
//   rhs     = ((threshold^2 >> 32) * norm_sq) >> 28
// ---------------------------------------------------------------------------
reg [51:0] dot_sq_s2;
reg [83:0] rhs_full_s2;
reg        dot_nonneg_s2;
reg        cmp_valid_s2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dot_sq_s2     <= 52'd0;
        rhs_full_s2   <= 84'd0;
        dot_nonneg_s2 <= 1'b0;
        cmp_valid_s2  <= 1'b0;
    end else begin
        cmp_valid_s2  <= cmp_valid_s1;
        dot_nonneg_s2 <= dot_nonneg_s1;

        if (cmp_valid_s1) begin
            dot_sq_s2   <= dot_sq_full_s1[95:44];
            rhs_full_s2 <= thr_sq_s1[63:32] * norm_sq_full_s1[95:44];
        end
    end
end

// ---------------------------------------------------------------------------
// 6. Compare pipeline stage 3
// ---------------------------------------------------------------------------
reg cmp_result;
reg cmp_valid;

wire [51:0] rhs_s3 = rhs_full_s2[79:28];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cmp_result <= 1'b0;
        cmp_valid  <= 1'b0;
    end else begin
        cmp_valid <= cmp_valid_s2;

        if (cmp_valid_s2) begin
            cmp_result <= dot_nonneg_s2 && (dot_sq_s2 >= rhs_s3);
        end
    end
end

// ---------------------------------------------------------------------------
// 7. Output register
// ---------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        match        <= 1'b0;
        result_valid <= 1'b0;
    end else begin
        result_valid <= cmp_valid;

        if (cmp_valid) begin
            match <= cmp_result;
        end
    end
end

endmodule