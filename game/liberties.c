
#include <glib.h>

typedef struct {
    int x;
    int y;
} Position;

static gboolean position_equal(gconstpointer a, gconstpointer b) {
    const Position *pos_a = a;
    const Position *pos_b = b;
    return pos_a->x == pos_b->x && pos_a->y == pos_b->y;
}

static guint position_hash(gconstpointer key) {
    const Position *pos = key;
    return g_direct_hash(GINT_TO_POINTER(pos->x)) ^ g_direct_hash(GINT_TO_POINTER(pos->y));
}

int count_liberties(int **matrix, int matrix_size, Position position) {
    GQueue *stack = g_queue_new();
    GHashTable *liberties_set = g_hash_table_new_full(position_hash, position_equal, NULL, NULL);
    GHashTable *visited = g_hash_table_new_full(position_hash, position_equal, NULL, NULL);

    int color = matrix[position.x][position.y];
    g_queue_push_head(stack, &position);

    Position directions[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    while (!g_queue_is_empty(stack)) {
        Position *current = g_queue_pop_tail(stack);
        if (g_hash_table_contains(visited, current)) {
            continue;
        }
        g_hash_table_add(visited, current);

        for (int i = 0; i < 4; i++) {
            Position next = {current->x + directions[i].x, current->y + directions[i].y};

            if (next.x < 0 || next.y < 0 || next.x >= matrix_size || next.y >= matrix_size) {
                continue;
            }

            if (matrix[next.x][next.y] == 0) {
                g_hash_table_add(liberties_set, &next);
            } else if (matrix[next.x][next.y] == color && !g_hash_table_contains(visited, &next)) {
                g_queue_push_head(stack, &next);
            }
        }
    }

    int n_unique_liberties = g_hash_table_size(liberties_set);

    g_queue_free(stack);
    g_hash_table_destroy(liberties_set);
    g_hash_table_destroy(visited);

    return n_unique_liberties;
}
