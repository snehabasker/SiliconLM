"""Writes the 5 benchmark problems (prompt, testbench, reference solution) to problems/*.json."""

import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "problems"
OUT.mkdir(exist_ok=True)

PROBLEMS = [
    {
        "pid": "adder4",
        "prompt": "Implement a 4-bit unsigned adder with carry out.\n"
                  "Module header:\nmodule adder4(input [3:0] a, input [3:0] b, "
                  "output [4:0] sum);\nsum must equal a + b (5-bit result). "
                  "Combinational logic only. Reply with complete Verilog.",
        "module_header": "module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);",
        "reference": "module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);\n"
                     "  assign sum = a + b;\nendmodule",
        "testbench": """
module tb;
  reg [3:0] a, b; wire [4:0] sum; integer i, j, errors;
  adder4 dut(.a(a), .b(b), .sum(sum));
  initial begin
    errors = 0;
    for (i = 0; i < 16; i = i + 1)
      for (j = 0; j < 16; j = j + 1) begin
        a = i[3:0]; b = j[3:0]; #1;
        if (sum !== (i + j)) begin
          errors = errors + 1;
          $display("FAIL a=%0d b=%0d sum=%0d", i, j, sum);
        end
      end
    if (errors == 0) $display("ALL_TESTS_PASSED");
    $finish;
  end
endmodule
""",
    },
    {
        "pid": "mux2",
        "prompt": "Implement a 2-to-1 multiplexer for 8-bit data.\n"
                  "Module header:\nmodule mux2(input [7:0] d0, input [7:0] d1, "
                  "input sel, output [7:0] y);\ny = d1 when sel is 1, else d0. "
                  "Reply with complete Verilog.",
        "module_header": "module mux2(input [7:0] d0, input [7:0] d1, input sel, output [7:0] y);",
        "reference": "module mux2(input [7:0] d0, input [7:0] d1, input sel, output [7:0] y);\n"
                     "  assign y = sel ? d1 : d0;\nendmodule",
        "testbench": """
module tb;
  reg [7:0] d0, d1; reg sel; wire [7:0] y; integer i, errors;
  mux2 dut(.d0(d0), .d1(d1), .sel(sel), .y(y));
  initial begin
    errors = 0;
    for (i = 0; i < 64; i = i + 1) begin
      d0 = $random; d1 = $random; sel = i[0]; #1;
      if (y !== (sel ? d1 : d0)) errors = errors + 1;
    end
    if (errors == 0) $display("ALL_TESTS_PASSED");
    $finish;
  end
endmodule
""",
    },
    {
        "pid": "dff_arst",
        "prompt": "Implement a D flip-flop with active-high asynchronous reset.\n"
                  "Module header:\nmodule dff_arst(input clk, input rst, input d, "
                  "output reg q);\nOn rst q becomes 0 immediately; otherwise q "
                  "captures d on the rising clock edge. Reply with complete Verilog.",
        "module_header": "module dff_arst(input clk, input rst, input d, output reg q);",
        "reference": "module dff_arst(input clk, input rst, input d, output reg q);\n"
                     "  always @(posedge clk or posedge rst)\n"
                     "    if (rst) q <= 1'b0; else q <= d;\nendmodule",
        "testbench": """
module tb;
  reg clk, rst, d; wire q; integer errors;
  dff_arst dut(.clk(clk), .rst(rst), .d(d), .q(q));
  always #5 clk = ~clk;
  initial begin
    errors = 0; clk = 0; rst = 1; d = 1; #12;
    if (q !== 1'b0) errors = errors + 1;      // held in reset
    rst = 0; d = 1; @(posedge clk); #1;
    if (q !== 1'b1) errors = errors + 1;      // captured d=1
    d = 0; @(posedge clk); #1;
    if (q !== 1'b0) errors = errors + 1;      // captured d=0
    d = 1; #2; rst = 1; #1;
    if (q !== 1'b0) errors = errors + 1;      // async reset mid-cycle
    if (errors == 0) $display("ALL_TESTS_PASSED");
    $finish;
  end
endmodule
""",
    },
    {
        "pid": "counter3",
        "prompt": "Implement a 3-bit up-counter with synchronous active-high reset "
                  "and enable.\nModule header:\nmodule counter3(input clk, input rst, "
                  "input en, output reg [2:0] count);\nOn the rising edge: reset to 0 "
                  "if rst, else increment when en. Wraps 7 to 0. Reply with complete Verilog.",
        "module_header": "module counter3(input clk, input rst, input en, output reg [2:0] count);",
        "reference": "module counter3(input clk, input rst, input en, output reg [2:0] count);\n"
                     "  always @(posedge clk)\n"
                     "    if (rst) count <= 3'd0;\n"
                     "    else if (en) count <= count + 3'd1;\nendmodule",
        "testbench": """
module tb;
  reg clk, rst, en; wire [2:0] count; integer i, errors; reg [2:0] model;
  counter3 dut(.clk(clk), .rst(rst), .en(en), .count(count));
  always #5 clk = ~clk;
  initial begin
    errors = 0; clk = 0; rst = 1; en = 0; model = 0;
    @(posedge clk); #1; rst = 0; en = 1;
    for (i = 0; i < 20; i = i + 1) begin
      @(posedge clk); model = model + 1; #1;
      if (count !== model) errors = errors + 1;
    end
    en = 0; @(posedge clk); #1;
    if (count !== model) errors = errors + 1;   // hold when disabled
    if (errors == 0) $display("ALL_TESTS_PASSED");
    $finish;
  end
endmodule
""",
    },
    {
        "pid": "prienc4",
        "prompt": "Implement a 4-to-2 priority encoder with valid flag.\n"
                  "Module header:\nmodule prienc4(input [3:0] in, output reg [1:0] out, "
                  "output reg valid);\nHighest set bit wins (bit 3 highest). valid is 1 "
                  "when any input bit is set, else 0 and out is 0. Reply with complete Verilog.",
        "module_header": "module prienc4(input [3:0] in, output reg [1:0] out, output reg valid);",
        "reference": "module prienc4(input [3:0] in, output reg [1:0] out, output reg valid);\n"
                     "  always @(*) begin\n"
                     "    valid = |in;\n"
                     "    casez (in)\n"
                     "      4'b1???: out = 2'd3;\n"
                     "      4'b01??: out = 2'd2;\n"
                     "      4'b001?: out = 2'd1;\n"
                     "      4'b0001: out = 2'd0;\n"
                     "      default: out = 2'd0;\n"
                     "    endcase\n"
                     "  end\nendmodule",
        "testbench": """
module tb;
  reg [3:0] in; wire [1:0] out; wire valid; integer i, errors;
  reg [1:0] exp; reg expv;
  prienc4 dut(.in(in), .out(out), .valid(valid));
  initial begin
    errors = 0;
    for (i = 0; i < 16; i = i + 1) begin
      in = i[3:0]; #1;
      expv = |in;
      if (in[3]) exp = 3; else if (in[2]) exp = 2;
      else if (in[1]) exp = 1; else exp = 0;
      if (valid !== expv) errors = errors + 1;
      else if (expv && out !== exp) errors = errors + 1;
    end
    if (errors == 0) $display("ALL_TESTS_PASSED");
    $finish;
  end
endmodule
""",
    },
]

for p in PROBLEMS:
    (OUT / f"{p['pid']}.json").write_text(json.dumps(p, indent=2))
print(f"wrote {len(PROBLEMS)} problems to {OUT}")
