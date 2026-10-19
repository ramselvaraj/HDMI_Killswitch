# ─── HDMI IN (TMDS) ───────────────────────────────────────
set_property -dict {PACKAGE_PIN H16 IOSTANDARD TMDS_33} [get_ports hdmi_rx_clk_p]
set_property -dict {PACKAGE_PIN H17 IOSTANDARD TMDS_33} [get_ports hdmi_rx_clk_n]
set_property -dict {PACKAGE_PIN D19 IOSTANDARD TMDS_33} [get_ports {hdmi_rx_data_p[0]}]
set_property -dict {PACKAGE_PIN D20 IOSTANDARD TMDS_33} [get_ports {hdmi_rx_data_n[0]}]
set_property -dict {PACKAGE_PIN B19 IOSTANDARD TMDS_33} [get_ports {hdmi_rx_data_p[1]}]
set_property -dict {PACKAGE_PIN A20 IOSTANDARD TMDS_33} [get_ports {hdmi_rx_data_n[1]}]
set_property -dict {PACKAGE_PIN B21 IOSTANDARD TMDS_33} [get_ports {hdmi_rx_data_p[2]}]
set_property -dict {PACKAGE_PIN B22 IOSTANDARD TMDS_33} [get_ports {hdmi_rx_data_n[2]}]

# ─── HDMI OUT (TMDS) ──────────────────────────────────────
set_property -dict {PACKAGE_PIN L16 IOSTANDARD TMDS_33} [get_ports hdmi_tx_clk_p]
set_property -dict {PACKAGE_PIN L17 IOSTANDARD TMDS_33} [get_ports hdmi_tx_clk_n]
set_property -dict {PACKAGE_PIN K17 IOSTANDARD TMDS_33} [get_ports {hdmi_tx_data_p[0]}]
set_property -dict {PACKAGE_PIN K18 IOSTANDARD TMDS_33} [get_ports {hdmi_tx_data_n[0]}]
set_property -dict {PACKAGE_PIN G17 IOSTANDARD TMDS_33} [get_ports {hdmi_tx_data_p[1]}]
set_property -dict {PACKAGE_PIN H18 IOSTANDARD TMDS_33} [get_ports {hdmi_tx_data_n[1]}]  
set_property -dict {PACKAGE_PIN E18 IOSTANDARD TMDS_33} [get_ports {hdmi_tx_data_p[2]}]
set_property -dict {PACKAGE_PIN E19 IOSTANDARD TMDS_33} [get_ports {hdmi_tx_data_n[2]}]

# ─── HDMI HPD (Hot Plug Detect) ───────────────────────────
set_property -dict {PACKAGE_PIN N18 IOSTANDARD LVCMOS33} [get_ports hdmi_rx_hpd]
set_property -dict {PACKAGE_PIN N17 IOSTANDARD LVCMOS33} [get_ports hdmi_tx_hpd]

# ─── Timing Constraints ───────────────────────────────────
# 720p pixel clock = 74.25 MHz
create_clock -period 13.468 -name hdmi_rx_clk [get_ports hdmi_rx_clk_p]