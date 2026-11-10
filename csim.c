// student id: 2025203019
// please change the above line to your student id

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <getopt.h>

struct cache_line
{
  uint64_t tag;
  uint32_t last_access_time;
  unsigned int valid : 1;
  unsigned int dirty : 1;
};

struct cache
{
  int s;                    /* set index 位数 (num_sets = 2^s) */
  int b;                    /* block offset 位数 (block_size = 2^b) */
  int assoc;                /* 每个 Set 的 Way 数量 (E) */
  int num_sets;             /* Set 数量 */
  int block_size;           /* 块大小  */

  uint64_t timer; /* 全局自增逻辑时钟 */

  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
  
  struct cache_line *lines; /* 一维扁平连续数组，大小为 num_sets * assoc */
};

/* 获取第 set_idx 组的首个 line 指针 */
static inline struct cache_line *
cache_get_set (const struct cache *c, int set_idx)
{
  return &c->lines[set_idx * c->assoc];
}

/* 从 64 位内存地址解析出 Set Index */
static inline int
cache_extract_set_index (const struct cache *c, uint64_t addr)
{
  if (c->s == 20)
    {
      return 0;
    }
  return (int)((addr >> c->b) & ((1u << c->s) - 1));
}

/* 从 64 位内存地址解析出 Tag */
static inline uint64_t
cache_extract_tag (const struct cache *c, uint64_t addr)
{
  return addr >> (c->s + c->b);
}

struct cache *
cache_create (int s, int assoc, int b)
{
  struct cache *c;
  size_t total_lines;

  c = malloc (sizeof (*c));
  if (c == NULL)
    {
      return NULL;
    }

  c->s = s;
  c->assoc = assoc;
  c->b = b;
  c->num_sets = 1 << s;
  c->block_size = 1 << b;
  c->timer = 0;
  c->hits = 0;
  c->misses = 0;
  c->evictions = 0;

  total_lines = (size_t)c->num_sets * (size_t)c->assoc;
  c->lines = calloc (total_lines, sizeof (struct cache_line));
  if (c->lines == NULL)
    {
      free (c);
      return NULL;
    }

  return c;
}

void
cache_destroy (struct cache *c)
{
  if (c != NULL)
    {
      free (c->lines);
      free (c);
    }
}

/* 在指定 set 中查找 tag 匹配的有效 line，命中返回 way 索引，未命中返回 -1 */
static int
cache_set_fint_hit (const struct cache_line *set_lines, int assoc,
                    uint64_t tag)
{
  int i;

  for (i = 0; i < assoc; i++)
    {
      if (set_lines[i].valid && set_lines[i].tag == tag)
        {
          return i;
        }
    }
  return -1;
}

/* 在指定 set 中寻找第一个空闲无效行，找不到返回 -1 */
static int
cache_set_find_empty (const struct cache_line *set_lines, int assoc)
{
  int i;

  for (i = 0; i < assoc; i++)
    {
      if (!set_lines[i].valid)
        {
          return i;
        }
    }
  return -1;
}

/* 在指定 set 中找到 last_access_time 最小的行（最久未访问）作为淘汰目标 */
static int
cache_set_find_victim (const struct cache_line *set_lines, int assoc)
{
  int i;
  int victim_idx = 0;
  uint64_t min_time = set_lines[0].last_access_time;

  for (i = 1; i < assoc; i++)
    {
      if (set_lines[i].last_access_time < min_time)
        {
          min_time = set_lines[i].last_access_time;
          victim_idx = i;
        }
    }
  return victim_idx;
}

/*
 * 访存统一入口函数
 * is_write: 1 为写操作（写回策略，标记 dirty），0 为读操作
 */
void
cache_access(struct cache *c, uint64_t addr, int8_t is_write)
{
  int set_idx;
  uint64_t tag;
  struct cache_line *set_lines;
  int hit_way;
  int target_way;

  c->timer++;
  set_idx = cache_extract_set_index (c, addr);
  tag = cache_extract_tag (c, addr);
  set_lines = cache_get_set (c, set_idx);

  hit_way = cache_set_fint_hit (set_lines, c->assoc, tag);

  if (hit_way != -1)
    {
      c->hits++;
      set_lines[hit_way].last_access_time = c->timer;
      if (is_write)
        {
          set_lines[hit_way].dirty = 1;
        }
      return;
    }

  c->misses++;

  target_way = cache_set_find_empty (set_lines, c->assoc);

  if (target_way == -1)
    {
      c->evictions++;
      target_way = cache_set_find_victim (set_lines, c->assoc);

      if (set_lines[target_way].dirty)
        {
          set_lines[target_way].dirty = 0;
        }
    }

  set_lines[target_way].valid = 1;
  set_lines[target_way].tag = tag;
  set_lines[target_way].last_access_time = c->timer;
  set_lines[target_way].dirty = is_write;
}    

void
print_summary(int hits, int misses, int evictions)
{
  printf("hits:%d misses:%d evictions:%d\n", hits, misses, evictions);
  FILE *output_fp = fopen(".csim_results", "w");
  assert(output_fp);
  fprintf(output_fp, "%d %d %d\n", hits, misses, evictions);
  fclose(output_fp);
}

void
print_help(const char *name)
{
  printf(
      "Usage: %s [-hv] -s <num> -E <num> -b <num> -t <file>\n"
      "Options:\n"
      "  -h         Print this help message.\n"
      "  -v         Optional verbose flag.\n"
      "  -s <num>   Number of set index bits.\n"
      "  -E <num>   Number of lines per set.\n"
      "  -b <num>   Number of block offset bits.\n"
      "  -t <file>  Trace file.\n\n"
      "Examples:\n"
      "  linux>  %s -s 4 -E 1 -b 4 -t traces/yi.trace\n"
      "  linux>  %s -v -s 8 -E 2 -b 4 -t traces/yi.trace\n",
      name, name, name);
}

int
main (int argc, char *argv[])
{
  int opt;
  int verbose = 0;
  int s = -1;
  int E = -1;
  int b = -1;
  char *tracefile = NULL;

  while ((opt = getopt (argc, argv, "hvs:E:b:t:")) != -1)
    {
      switch (opt)
        {
        case 'h':
          print_help (argv[0]);
          exit (0);
        case 'v':
          verbose = 1;
          break;
        case 's':
          s = atoi(optarg);
          break;
        case 'E':
          E = atoi(optarg);
          break;
        case 'b':
          b = atoi(optarg);
          break;
        case 't':
          tracefile = optarg;
          break;
        default:
          print_help(argv[0]);
          exit(1);
        }
    }


  if (s <= 0 || E <= 0 || b <= 0 || tracefile == NULL)
    {
      fprintf(stderr, "%s: Missing required command line argument\n", argv[0]);
      print_help (argv[0]);
      exit (1);
    }

  struct cache *sim = cache_create (s, E, b);
  if (!sim)
    {
      fprintf (stderr, "Failed to allocate cache\n");
      exit(1);
    }
  
  FILE *trace_fp = fopen (tracefile, "r");
  if (trace_fp == NULL)
    {
      perror (tracefile);
      cache_destroy (sim);
      return 1;
    }

  char line[256];
  while (fgets (line, sizeof (line), trace_fp) != NULL)
    {
      char operation;
      unsigned long long address;
      int size;

      if (sscanf (line, " %c %llx,%d", &operation, &address, &size) != 3)
        {
          continue;
        }
      if (operation != 'L' && operation != 'S')
        {
          continue;
        }

      uint64_t old_hits = sim->hits;
      uint64_t old_evictions = sim->evictions;

      cache_access (sim, (uint64_t)address, operation == 'S');

      if (verbose)
        {
          printf ("%c %llx,%d %s", operation, address, size,
                  sim->hits > old_hits ? "hit" : "miss");
          if (sim->evictions > old_evictions)
            {
              printf (" eviction");
            }
          printf ("\n");
        }
    }

  if (ferror (trace_fp))
    {
      perror (tracefile);
      fclose (trace_fp);
      cache_destroy (sim);
      return 1;
    }

  fclose (trace_fp);
  print_summary (sim->hits, sim->misses, sim->evictions);
  cache_destroy (sim);

  return 0;
}
