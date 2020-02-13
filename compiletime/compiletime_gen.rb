# syclbench compile time benchmarking generator script

# generates SYCL programs of various size and structure
# run "ruby compiletime_gen.rb -h" for documentation

DECL_INC_FILE = "kernel_declarations.inc"
KERNEL_INC_FILE = "kernels.inc"

RT_SIZE = "rt_size"

# operations available for generation

OP_MAPPING = {
        "sin" => "OUT = cl::sycl::sin(IN1);",
        "cos" => "OUT = cl::sycl::cos(IN1);",
        "sqrt" => "OUT = cl::sycl::sqrt(IN1);",
        "add" => "OUT = IN1 + IN2;",
        "mad" => "OUT = IN1 * IN2 + IN1;",
}

TYPES = %i[int float double]

# PARAMETERS

require 'optparse'
require 'ostruct'
require 'pp'


def parse_cmd(args)
        # defaults
        options = OpenStruct.new
        options.num_kernels = 1
        options.num_buffers = 2
        options.num_captures = 4
        options.dimensions = 1
        options.loopnests = 1
        options.type = :float
        options.mix = [["mad",10]]
        options.templated = false
        options.verbose = false

        opt_parser = OptionParser.new do |opts|
                opts.banner = "Usage: compiletime_gen.rb [options]"

                opts.on("-k", "--num_kernels NUM", Integer, "Create NUM kernels") do |num|
                        options.num_kernels = num
                end
                opts.on("-b", "--num_buffers NUM", Integer, "Create NUM buffers") do |num|
                        options.num_buffers = num.clamp(1,1024*16) # need at least one buffer
                end
                opts.on("-c", "--num_captures NUM", Integer, "Use NUM captures") do |num|
                        options.num_captures = num
                end
                opts.on("-d", "--dimensions DIM", Integer, "Select dimensionality") do |dim|
                        options.dimensions = dim.clamp(1,3)
                end
                opts.on("-l", "--loopnests NUM", Integer, "Create NUM loop nests") do |num|
                        options.loopnests = num
                end
                opts.on("-t", "--type TYPE", TYPES, "Select data type (#{TYPES.join(',')})") do |type|
                        options.type = type
                end
                opts.on("-m", "--mix a,b,c", Array, "composition, e.g. sin:3,mad:10") do |mix|
                        options.mix = mix.map do |s| 
                                p = s.split(":")
                                [p[0], p[1].to_i]
                        end
                end
                opts.on("-T", "--[no-]templated", "Generate templated kernels") do |v|
                        options.templated = v
                end

                opts.on("-v", "--[no-]verbose", "Run verbosely") do |v|
                        options.verbose = v
                end
                opts.on_tail("-h", "--help", "Show this message") do
                        puts opts
                        exit
                end
        end
        opt_parser.parse!(args)
        p options.mix
        options
end

options = parse_cmd(ARGV)
#pp options

kernel_names = 1.upto(options.num_kernels).to_a.map { |i| "kernel_#{i}" }

buffer_names = 1.upto(options.num_buffers).to_a.map { |i| "buffer_#{i}" }
acc_names = buffer_names.map { |n| n + "_acc" }
buffers = buffer_names.zip(acc_names)

capture_names = 1.upto(options.num_captures).to_a.map { |i| "capture_#{i}" }

File.open(DECL_INC_FILE, "w+") do |f|
        kernel_names.each do |kn|
                decl = ""
                decl += "template <typename _TT, int _TN> " if options.templated
                decl += "class #{kn};"
                f.puts decl
        end
        f.puts
end


File.open(KERNEL_INC_FILE, "w+") do |f|

        spaces = 0
        fwr = ->(str) { # print with indentation (for no real reason)
                newspaces = spaces + (str.count("{") - str.count("}")) * 2
                spaces = newspaces if newspaces < spaces
                f.puts(" "*spaces + str)
                spaces = newspaces
        }

        ndrange = Array.new(options.dimensions, RT_SIZE).join(",")

        buffer_names.each do |bn|
                fwr.call "s::buffer<#{options.type}, #{options.dimensions}> #{bn}{s::range<#{options.dimensions}>(#{ndrange})};"
        end
        f.puts

        capture_names.each do |cn|
                fwr.call "#{options.type} #{cn}{};"
        end
        f.puts

        kernel_names.each do |kn|
                fwr.call "device_queue.submit([&](cl::sycl::handler& cgh) {"

                buffers.each do |bn, an|
                        fwr.call "auto #{an} = #{bn}.get_access<s::access::mode::read_write>(cgh);"
                end

                fwr.call "cl::sycl::range<#{options.dimensions}> ndrange{#{ndrange}};"

                full_kernel_name = kn
                full_kernel_name += "<#{options.type}, #{otions.dimensions}>" if options.templated
                fwr.call "cgh.parallel_for<#{full_kernel_name}>(ndrange, [=](cl::sycl::id<#{options.dimensions}> gid) {"
                fwr.call "#{acc_names[0]}[gid] += #{capture_names.join(" + ")};" # use each capture
                fwr.call "#{acc_names[0]}[gid] += #{acc_names.join("[gid] + ")}[gid];" # use each buffer


                acc_idx = 0
                next_acc = -> { acc_idx = (acc_idx+1)%options.num_buffers; acc_names[acc_idx] }

                options.loopnests.times do |i|
                        fwr.call "for(#{options.type} i#{i} = 0; i#{i} < #{next_acc.call}[gid]; ++i#{i}) {"
                end

                options.mix.each do |i_type, count|
                        count.times do
                                ins = OP_MAPPING[i_type]
                                ins.gsub!("OUT", "#{next_acc.call}[gid]")
                                ins.gsub!("IN1", "#{next_acc.call}[gid]")
                                ins.gsub!("IN2", "#{next_acc.call}[gid]")
                                fwr.call "#{ins}"
                        end
                end

                options.loopnests.times do
                        fwr.call "}"
                end

                fwr.call "}); // parallel_for"
                fwr.call "}); // submit\n\n"
        end
end