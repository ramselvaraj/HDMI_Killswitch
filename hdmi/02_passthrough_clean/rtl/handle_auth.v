module handle_auth (
    input        auth,
    input  [23:0] vid_pData_in,
    output [23:0] vid_pData_out
);

assign vid_pData_out = auth ? vid_pData_in : 24'h000000;

endmodule